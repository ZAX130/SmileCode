#include<torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_3DFEATMAP(height, width, length, kernel_size) TORCH_CHECK(height >= kernel_size && width >= kernel_size && length >= kernel_size, "Input resolution must be greater than or equal to kernel size.")
#define CHECK_KERNELSIZE(NAME, kernel_size) TORCH_CHECK( \
        kernel_size == 3 || kernel_size == 5 || kernel_size == 7 || \
        kernel_size == 9 || kernel_size == 11 || kernel_size == 13, \
        NAME, " does not support kernel size ", kernel_size)
#define CUDA_NUM_THREADS 1024

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block=CUDA_NUM_THREADS) {
  auto block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int>(block_num);
}

torch::Tensor modet_fw_cu(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb
);

inline __host__ __device__ int get_backward_window_start(const int index, const int KERNEL_SIZE)
{
    return (index + 1 < KERNEL_SIZE) ? 0 : index - KERNEL_SIZE + 1;
}


inline __host__ __device__ int get_backward_window_end(const int index, const int length, const int KERNEL_SIZE)
{
    return (index + KERNEL_SIZE >= length) ? length - KERNEL_SIZE : index;
}
std::vector<torch::Tensor> modet_bw_cu(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled
);