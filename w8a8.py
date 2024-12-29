import triton
import triton.language as tl
import torch

@triton.jit
def _w8a16_mm_kernel(
    # 指针参数
    input_ptr, weight_ptr, output_ptr,
    input_scale_ptr, weight_scale_ptr,
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
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wm + offs_n[None, :] * stride_wn
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # 加载scale
    input_scale = tl.load(input_scale_ptr + offs_m)
    weight_scale = tl.load(weight_scale_ptr + offs_n)

    # 主循环
    for k in range(0, K, BLOCK_K):
        # 加载输入和权重块
        input_block = tl.load(input_ptrs, mask=offs_m[:, None] < M and (k + offs_k[None, :]) < K, other=0)
        weight_block = tl.load(weight_ptrs, mask=(k + offs_k[:, None]) < K and offs_n[None, :] < N, other=0)
        
        # 量化输入
        input_block = tl.cast(input_block / input_scale[:, None], tl.int8)
        
        # 矩阵乘法
        acc += tl.dot(input_block, weight_block)
    
    # 反量化
    acc = acc.to(tl.float16)
    acc = acc * (input_scale[:, None] * weight_scale[None, :])

    # 存储结果
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, acc, mask=offs_m[:, None] < M and offs_n[None, :] < N)

def w8a16_mm(input, weight, input_scale, weight_scale):
    # 获取维度
    M, K = input.shape
    K, N = weight.shape

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
    _w8a16_mm_kernel[grid](
        input, weight, output,
        input_scale, weight_scale,
        M, N, K,
        stride_im, stride_in,
        stride_wm, stride_wn,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return output
