import triton
import triton.language as tl
import torch

@triton.jit
def _dds_mm_kernel(
    # 指针参数
    input_ptr, value_ptr, output_ptr,
    row_offset_ptr, col_indice_ptr,
    # 矩阵维度参数
    M, N, K,
    # BSR相关参数
    BLOCK_SIZE: tl.constexpr,
    # 步长参数
    stride_im, stride_in,  # input的步长
    stride_om, stride_on,  # output的步长
    # kernel块大小参数
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 计算程序ID和网格维度
    pid = tl.program_id(0)
    
    # 将维度转换为正确的类型
    M = tl.load(M)  # 转换为标量
    N = tl.load(N)  # 转换为标量
    
    # 计算网格维度
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    # 计算二维索引
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 计算当前块的行偏移
    block_row_start = pid_m * BLOCK_M // BLOCK_SIZE
    block_row_end = (pid_m * BLOCK_M + BLOCK_M + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 计算输出偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 遍历每一个非零块
    for block_row in range(block_row_start, block_row_end):
        # 加载行偏移
        row_start = tl.load(row_offset_ptr + block_row)
        row_end = tl.load(row_offset_ptr + block_row + 1)

        # 遍历该行中的所有非零块
        for block_idx in range(row_start, row_end):
            # 加载列索引
            block_col = tl.load(col_indice_ptr + block_idx)
            
            # 计算当前块在原矩阵中的位置
            row_idx = block_row * BLOCK_SIZE
            col_idx = block_col * BLOCK_SIZE

            # 加载输入块 - 修改为加载BLOCK_M x BLOCK_SIZE大小的块
            input_block = tl.load(
                input_ptr + offs_m[:, None] * stride_im + (col_idx + tl.arange(0, BLOCK_SIZE)[None, :]) * stride_in,
                mask=mask_m,
                other=0.0
            )

            # 加载权重块 - 修改为加载BLOCK_SIZE x BLOCK_N大小的块
            weight_block = tl.load(
                value_ptr + block_idx * BLOCK_SIZE * BLOCK_SIZE + 
                tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE +
                tl.arange(0, BLOCK_SIZE)[None, :],
                mask=tl.arange(0, BLOCK_SIZE)[:, None] < BLOCK_SIZE,
                other=0.0
            )

            # 执行矩阵乘法 - 确保维度匹配：(BLOCK_M x BLOCK_SIZE) @ (BLOCK_SIZE x BLOCK_N)
            acc += tl.dot(
                input_block.to(tl.float32),  # shape: (BLOCK_M, BLOCK_SIZE)
                weight_block.to(tl.float32)   # shape: (BLOCK_SIZE, BLOCK_N)
            )

    # 存储结果
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, acc.to(tl.float16), mask=mask_m & mask_n)

def dds_mm(input, shape, value, row_offset, col_indice):
    """
    执行dense x sparse矩阵乘法
    """
    M, K = input.shape
    K, N = shape
    
    # 确保输入是fp16类型
    assert input.dtype == torch.float16
    assert value.dtype == torch.float16
    
    # 将维度转换为tensor
    M = torch.tensor([M], device=input.device)
    N = torch.tensor([N], device=input.device)
    K = torch.tensor([K], device=input.device)
    
    # 计算步长
    stride_im, stride_in = input.stride(0), input.stride(1)
    
    # 分配输出空间
    output = torch.empty((M.item(), N.item()), device=input.device, dtype=torch.float16)
    stride_om, stride_on = output.stride(0), output.stride(1)

    # BSR块大小
    BLOCK_SIZE = 32
    
    # kernel块大小 - 修改为与BSR块大小匹配
    BLOCK_M = 32  # 修改为与BLOCK_SIZE相同
    BLOCK_N = 32  # 修改为与BLOCK_SIZE相同

    # 计算网格大小
    grid = lambda META: (triton.cdiv(N.item(), META['BLOCK_N']) * triton.cdiv(M.item(), META['BLOCK_M']),)

    # 启动kernel
    _dds_mm_kernel[grid](
        input, value, output,
        row_offset, col_indice,
        M, N, K,
        BLOCK_SIZE,
        stride_im, stride_in,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return output 