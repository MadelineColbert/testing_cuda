#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdio>
#include "include/em_mm_types.h"

using data_type= double;

size_response test_size_shenans(
    const std::vector<data_type> A, 
    const std::vector<data_type> B, 
    const std::vector<data_type> C) 
{
    float m = sqrt(A.size()*B.size()/C.size());
    int s = (int)m;
    if (s != m) {
        return {.sz=-1, .error_code=SIZE_MISMATCH};
    }
    return {.sz=s, .error_code=MMSUCCESS};
}

MMStatus matrix_multiplication(
    std::vector<data_type>& A, 
    std::vector<data_type>& B, 
    std::vector<data_type>& C
    )
{
    size_response c_ex_resp = test_size_shenans(A, B, C);
    size_response b_ex_resp = test_size_shenans(A, C, B);
    size_response a_ex_resp = test_size_shenans(C, B, A);

    if (
        c_ex_resp.error_code == SIZE_MISMATCH |
        b_ex_resp.error_code == SIZE_MISMATCH |
        a_ex_resp.error_code == SIZE_MISMATCH
    ) {
        return SIZE_MISMATCH;
    }

    int c_ex = c_ex_resp.sz;
    int b_ex = b_ex_resp.sz;
    int a_ex = a_ex_resp.sz; 

    cublasHandle_t h = NULL;
    cudaStream_t stream = NULL;
    cublasOperation_t op1 = CUBLAS_OP_N;
    cublasOperation_t op2 = CUBLAS_OP_N;
    data_type *d_a = nullptr;
    data_type *d_b = nullptr;
    data_type *d_c = nullptr;

    cublasCreate(&h);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(h, stream);

    cudaMalloc((void**)&d_a, b_ex*c_ex*sizeof(data_type));
    cudaMalloc((void**)&d_b, c_ex*a_ex*sizeof(data_type));
    cudaMalloc((void**)&d_c, b_ex*a_ex*sizeof(data_type));

    cudaMemcpyAsync(d_a, A.data(), b_ex*c_ex*sizeof(data_type), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, B.data(), c_ex*a_ex*sizeof(data_type), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_c, C.data(), b_ex*a_ex*sizeof(data_type), cudaMemcpyHostToDevice, stream);

    const double alpha = 1.0;
    const double beta = 1.0;
    int lda = c_ex;
    int ldb = b_ex;
    int ldc = c_ex;

    cublasStatus_t status = cublasDgemm(
        h, op1, op2, c_ex, 
        a_ex, b_ex, &alpha, 
        d_a, lda, d_b, ldb, 
        &beta, d_c, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUDA error %d at %s:%d\n", status, __FILE__, __LINE__);
    }

    cudaMemcpyAsync(C.data(), d_c, b_ex*a_ex*sizeof(data_type), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cublasDestroy(h);
    cudaStreamDestroy(stream);

    return MMSUCCESS;
}

int main(int argc, char *argv[]) {
    std::vector<data_type> A = {1,2,3,4,5,6}; // 2x3
    std::vector<data_type> B = {1,2,3,4,5,6,1,2,3,4,5,6}; // 3x4
    std::vector<data_type> C = {1,2,3,4,5,6,7,8}; // 2x4
    
    MMStatus result = matrix_multiplication(A,B,C);
   
    printf("%d", result);

    for (auto i : C) {
        printf("%f ", i);
    }
    return 0;
    // printf("CEX: %d\n", c_ex);
    // printf("BEX: %d\n", b_ex);
    // printf("AEX: %d\n", a_ex);

}