import triton
import triton.language as tl
import torch

@triton.jit
def _w4a16_mm_kernel(
    # 指针参数
    input_ptr, weight_ptr, output_ptr,
    weight_scale_ptr,
    # 矩阵维度参数 
    M, N, K,
    # 步长参数
    stride_im, stride_in,  # input的步长
    stride_wm, stride_wn,  # weight的步长
    stride_om, stride_on,  # output的步长
    # 块大小参数
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 计算程序ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 创建指针
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_in
    # 由于8个INT4打包成一个INT32，所以K维度需要除以8
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wm // 8 + offs_n[None, :] * stride_wn
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 加载weight scale
    weight_scale = tl.load(weight_scale_ptr + offs_n)

    # 主循环
    for k in range(0, K, BLOCK_K):
        # 加载输入块
        input_block = tl.load(input_ptrs, mask=offs_m[:, None] < M and (k + offs_k[None, :]) < K, other=0)
        
        # 加载打包的权重块
        packed_weight = tl.load(weight_ptrs, mask=(k + offs_k[:, None]) < K and offs_n[None, :] < N, other=0)
        
        # 解包INT4权重 (每个INT32包含8个INT4)
        # 使用移位和掩码操作提取每个INT4，并确保得到有符号数
        weight_block = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.int32)
        for i in range(8):
            # 提取每个INT4并转换为有符号数
            shift = i * 4
            mask = 0xF
            values = (packed_weight >> shift) & mask
            # 将大于7的值转换为负数 (8->-8, 9->-7, ..., 15->-1)
            values = tl.where(values > 7, values - 16, values)
            weight_block = tl.where((offs_k[:, None] % 8) == i, values, weight_block)
        
        # 反量化权重
        weight_block = weight_block.to(tl.float32) * weight_scale[None, :]
        
        # 矩阵乘法
        acc += tl.dot(input_block.to(tl.float32), weight_block)
    
    # 转换为float16并存储结果
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < M and offs_n[None, :] < N)

def w4a16_mm(input, weight, weight_scale):
    # 获取维度
    M, K = input.shape
    K_div_8, N = weight.shape  # weight已经被打包，所以K维度是原来的1/8
    K = K_div_8 * 8

    # 计算步长
    stride_im, stride_in = input.stride(0), input.stride(1)
    stride_wm, stride_wn = weight.stride(0), weight.stride(1)
    
    # 分配输出空间
    output = torch.empty((M, N), device=input.device, dtype=torch.float16)
    stride_om, stride_on = output.stride(0), output.stride(1)

    # 定义块大小
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32

    # 计算网格大小
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    # 启动kernel
    _w4a16_mm_kernel[grid](
        input, weight, output,
        weight_scale,
        M, N, K,
        stride_im, stride_in,
        stride_wm, stride_wn,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return output
