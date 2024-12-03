#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>
#include<iostream>
#include "utils.h"
#include <stdio.h>
#define CUDA_NUM_THREADS_Q 512
#define CUDA_NUM_THREADS_K 512
#define CUDA_NUM_THREADS_RPB 64

template <typename scalar_t>
__global__ void modet_fw_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 6, torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t, 6, torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t, 6, torch::DefaultPtrTraits> attn,
    const int height,
    const int width,
    const int length,
    const int batch_size,
    const int heads,
    const int dim,
    const int KERNEL_SIZE)
{
    const int z = blockIdx.z * blockDim.z + threadIdx.z; // batch*heads
    if (z < batch_size * heads)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x; // batch*heads
        if (x < height * width * length)
        {
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE * KERNEL_SIZE* KERNEL_SIZE)
            {
                const int b = z / heads;     // 第几个batch
                const int h = z - b * heads; // 第几个batch中的第几个head

                // 计算3x3x3核上的第几个数
                int indtmp1 = y/KERNEL_SIZE;
                const int kk = y - indtmp1 * KERNEL_SIZE;

                int indtmp2 = indtmp1/KERNEL_SIZE;
                const int kj = indtmp1 - indtmp2 * KERNEL_SIZE;
                
                indtmp1 = indtmp2;
                indtmp2 = indtmp1/KERNEL_SIZE;
                const int ki = indtmp1 - indtmp2 * KERNEL_SIZE;

                // const int ki = y / (KERNEL_SIZE * KERNEL_SIZE);
                // const int kj = (y - ki * KERNEL_SIZE * KERNEL_SIZE) / KERNEL_SIZE;
                // const int kk = y - ki * KERNEL_SIZE * KERNEL_SIZE - kj * KERNEL_SIZE;

                // 计算要算第几个像素
                indtmp1 = x/length;
                const int k = x - indtmp1 * length;

                indtmp2 = indtmp1/width;
                const int j = indtmp1 - indtmp2 * width;

                indtmp1 = indtmp2;
                indtmp2 = indtmp1/height;
                const int i = indtmp1 - indtmp2 * height;
                // const int i = x / (width * length);
                // const int j = (x - i * width * length) / length;
                // const int k = x - i * width * length - j * length;
                // query:B,heads,H,W,L,dims
                // key:B,heads,H+2,W+2,L+2,dims
                // rpb:heads, kernel_size, kernel_size,kernel_size
                scalar_t updt = scalar_t(0);
                const int queryOffset = b * query.stride(0) + h * query.stride(1) + i * query.stride(2) + j * query.stride(3) + k * query.stride(4);
                const int keyOffset = b * key.stride(0) + h * key.stride(1) + (ki + i) * key.stride(2) + (kj + j) * key.stride(3) + (kk + k) * key.stride(4);
                #pragma unroll
                for (int dimOffset = 0; dimOffset < dim; dimOffset++)
                    updt += query.data()[queryOffset + dimOffset] * key.data()[keyOffset + dimOffset]; // q，k每个dim相乘再相加
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + k * attn.stride(4) + y * attn.stride(5);
                const int rpbIndex = h * rpb.stride(0) + ki * rpb.stride(1) + kj * rpb.stride(2) + kk * rpb.stride(3);
                updt += rpb.data()[rpbIndex];
                attn.data()[index] = updt;
            }
        }
    }
}

torch::Tensor modet_fw_cu(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb)
{ // query:B,heads,H,W,L,dims
    // key:B,heads,H+2,W+2,L+2,dims
    // rpb:heads, kernel_size, kernel_size,kernel_size
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t length = query.size(4);
    int64_t dim = query.size(5);
    int64_t RPB_MAX = rpb.size(1);
    int kernel_size = RPB_MAX;
    assert(kernel_size == 3);
    int kernel_size_sq = pow(kernel_size, 3);
    int zsize = batch_size * heads;
    int xsize = height * width * length;
    CHECK_3DFEATMAP(height, width, length, kernel_size);
    CHECK_KERNELSIZE("modet_fw_cu", kernel_size);

    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size_sq);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
        {batch_size, heads, height, width, length, kernel_size_sq}, query.options());
    // auto attn = torch::zeros(
    //         {6, 6, 12, 12, 12, 9}, query.options());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
        (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
        (kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS,
        (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    //     const dim3 blocks(16,16,16);
    // const dim3 threads(16, 16, 4);
    // std::cout<<"threads:"<<PIXELTHREADS<<' '<<KERNELTHREADS<<' '<<BATCHTHREADS<<std::endl;
    // std::cout<<"threads:"<<(xsize + PIXELTHREADS - 1) / PIXELTHREADS<<' '<<(kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS<<' '<<(zsize + BATCHTHREADS - 1) / BATCHTHREADS<<std::endl;
    // std::cout<<query.size(0)<<query.size(1)<<' '<<query.size(2)<<' '<<query.size(3)<<' '<<query.size(4)<<' '<<query.size(5)<<std::endl;
    // std::cout<<key.size(0)<<key.size(1)<<' '<<key.size(2)<<' '<<key.size(3)<<' '<<key.size(4)<<' '<<key.size(5)<<std::endl;
    // std::cout<<rpb.size(0)<<rpb.size(1)<<' '<<rpb.size(2)<<' '<<rpb.size(3)<<std::endl;
    // std::cout<<attn.size(0)<<attn.size(1)<<' '<<attn.size(2)<<' '<<attn.size(3)<<' '<<attn.size(4)<<' '<<attn.size(5)<<std::endl;
    // std::cout<<height<<' '<<width<<' '<<length<<' '<<batch_size<<' '<<heads<<' '<<dim<<std::endl;
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "modet_fw_cu", ([&]
                                                                    {
        const auto query_a = query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
         modet_fw_kernel<scalar_t><<<blocks, threads,0,stream>>>(
        query_a, // 形态转换
        key_a,
        rpb_a,
        attn_a,
        height,
        width,
        length,
        batch_size,
        heads,
        dim,
        kernel_size
    ); }));
    return attn;
}

template <typename scalar_t>
__global__ void modetdq_bw_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits> key,
    const int height,
    const int width,
    const int length, 
    const int heads,
    const int dim,
    const int totalElements,
    const int KERNEL_SIZE) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalElements){

        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;

        int indtmp2 = indtmp1/length;
        const int k = indtmp1 - indtmp2 * length;

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;

        scalar_t d_query_update = scalar_t(0);
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3) + k * d_attn.stride(4);
        const int keyOffset = b * key.stride(0) + h * key.stride(1) + d;
        #pragma unroll
        for (int xi=0; xi < KERNEL_SIZE; xi+=1)
            #pragma unroll
            for (int xj=0; xj < KERNEL_SIZE; xj+=1)
                #pragma unroll
                for (int xk=0; xk < KERNEL_SIZE; xk+=1){
                    const int keyIndex = keyOffset + (xi+i) * key.stride(2) + (xj+j) * key.stride(3) + (xk+k) * key.stride(4);
                    d_query_update += d_attn.data()[attnOffset] * key.data()[keyIndex];
                    ++attnOffset;
                }
        d_query.data()[linearIndex] = d_query_update;
        // printf("index=%d h=%d i=%d j=%d k=%d ni=%d nj=%d nk=%d ei=%d ej=%d ek=%d\n",linearIndex,h,i,j,k,ni,nj,nk,ei,ej,ek);
    }
}

template <typename scalar_t>
__global__ void modetdk_bw_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits> query,
    const int height,
    const int width,
    const int length,
    const int heads,
    const int dim,
    const int d_key_numel,
    const int KERNEL_SIZE) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_key_numel){
        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;

        int indtmp2 = indtmp1/length;
        const int k = indtmp1 - indtmp2 * length;

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;

        const int b = indtmp2;
        //算最先和最后包含k[i,j,k]的窗的q的ijk
        const int ni = get_backward_window_start(i, KERNEL_SIZE);
        const int nj = get_backward_window_start(j, KERNEL_SIZE);
        const int nk = get_backward_window_start(k, KERNEL_SIZE);
        const int ei = get_backward_window_end(i, height, KERNEL_SIZE);
        const int ej = get_backward_window_end(j, width, KERNEL_SIZE);
        const int ek = get_backward_window_end(k, length, KERNEL_SIZE);
        const int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1);
        const int queryOffset = b * query.stride(0) + h * query.stride(1) + d;
        
        scalar_t d_key_update = scalar_t(0);
        #pragma unroll
        for (int xi=ni; xi <= ei; xi+=1)
            #pragma unroll
            for (int xj=nj; xj <= ej; xj+=1)
                #pragma unroll
                for (int xk=nk; xk <= ek; xk+=1){
                    const int queryIndex = queryOffset + xi * query.stride(2) + xj * query.stride(3) + xk * query.stride(4);
                    const int attnIndex = attnOffset + xi * d_attn.stride(2) + xj * d_attn.stride(3) + xk * d_attn.stride(4) 
                    + (i-xi)*KERNEL_SIZE*KERNEL_SIZE + (j-xj)*KERNEL_SIZE + k-xk;
                    d_key_update += query.data()[queryIndex] * d_attn.data()[attnIndex];
                }
        d_key.data()[linearIndex] = d_key_update;
        
    }
}

template <typename scalar_t>
__global__ void modetdrpb_bw_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits> d_attn,
    const int height,
    const int width,
    const int length,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads,
    const int KERNEL_SIZE) {
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalThreads){

        int indtmp1 = linearIndex/KERNEL_SIZE;
        const int kk = linearIndex - indtmp1 * KERNEL_SIZE;

        int indtmp2 = indtmp1/KERNEL_SIZE;
        const int kj = indtmp1 - indtmp2 * KERNEL_SIZE;
        
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/KERNEL_SIZE;
        const int ki = indtmp1 - indtmp2 * KERNEL_SIZE;
        
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/length;
        const int k = indtmp1 - indtmp2 * length;

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;

        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        const int h = indtmp2;
        
        scalar_t d_rpb_update = scalar_t(0);
        int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3)+ k * d_attn.stride(4) + (ki*KERNEL_SIZE*KERNEL_SIZE + kj*KERNEL_SIZE + kk);
        #pragma unroll
        for (int b=0; b < batch_size; ++b){//在b维上累加
            d_rpb_update += d_attn.data()[attnOffset];
            attnOffset += d_attn.stride(0);
        }
        // printf("%d %d %d %d %d %d %d %d\n",linearIndex,h,i,j,k,ki,kj,kk);
        const int index = h * d_rpb.stride(0) + ki * d_rpb.stride(1) + kj * d_rpb.stride(2) + kk * d_rpb.stride(3);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}
std::vector<torch::Tensor> modet_bw_cu(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t length = query.size(4);
    int64_t dim = query.size(5);
    int64_t height_k = key.size(2);
    int64_t width_k = key.size(3);
    int64_t length_k = key.size(4);
    int kernel_size_sq = d_attn.size(5);
    
    int kernel_size = pow(kernel_size_sq, 1/3.0);

    // std::cout<<kernel_size_sq<<kernel_size<<std::endl;
    CHECK_3DFEATMAP(height, width,length, kernel_size);
    CHECK_KERNELSIZE("modet_backward", kernel_size);
    int64_t RPB_MAX = kernel_size;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX, RPB_MAX, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * height * width * length * kernel_size_sq;
    int blocks_rpb = GET_BLOCKS(n_rpb, CUDA_NUM_THREADS_RPB);
    dim3 grid_rpb(blocks_rpb);
    dim3 blockr(CUDA_NUM_THREADS_RPB);

    int32_t n_query = d_query.numel();
    int blocks_query = GET_BLOCKS(n_query, CUDA_NUM_THREADS_Q);
    dim3 grid_query(blocks_query);
    dim3 blockq(CUDA_NUM_THREADS_Q);

    int32_t n_key = d_key.numel();
    int blocks_key = GET_BLOCKS(n_key, CUDA_NUM_THREADS_K);
    dim3 grid_key(blocks_key);
    dim3 blockk(CUDA_NUM_THREADS_K);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "modet_bw_cu", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            modetdrpb_bw_kernel<scalar_t><<<grid_rpb, blockr, 0, stream>>>(
                    d_rpb_a, d_attn_a, height, width,length, batch_size, d_rpb.numel(), n_rpb, kernel_size);
        }
        modetdq_bw_kernel<scalar_t><<<grid_query, blockq, 0, stream>>>(
            d_query_a, d_attn_a, key_a, height, width, length, heads, dim, n_query,kernel_size);
        modetdk_bw_kernel<scalar_t><<<grid_key, blockk, 0, stream>>>(
            d_key_a, d_attn_a, query_a, height_k, width_k, length_k, heads, dim, n_key, kernel_size);
    }));
    return {d_query,d_key,d_rpb};
}