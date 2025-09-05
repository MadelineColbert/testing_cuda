#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <cstdio>
#include <random>
#include "include/common.h"

using data_type=double;

struct csp_resp {
    int m;
    int n;
    std::vector<data_type> h_A;
    std::vector<data_type> workspace;
    std::vector<int> devipiv;
    int devinfo=0;

    csp_resp(int mv, int nv) 
        : m(mv), n(nv),
          h_A(mv*nv),
          devipiv(min(mv,nv)) {

    }
};

void cusolver_problem(csp_resp *test_in) {
    cusolverDnHandle_t handle;
    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

    int lda=max(1,test_in->m);

    data_type *d_A = nullptr;
    data_type *d_workspace = nullptr;
    int *d_devipiv = nullptr;
    int *d_devinfo = nullptr;

    int malloc_size = test_in->m*test_in->n*sizeof(data_type);

    CUDA_CHECK(cudaMalloc((void**)&d_A, malloc_size));
    CUDA_CHECK(cudaMalloc((void**)&d_devipiv,  min(test_in->m,test_in->n)*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_devinfo,  sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, test_in->h_A.data(), malloc_size, cudaMemcpyHostToDevice, stream));

    int lwork=0;

    cusolverDnDgetrf_bufferSize(handle, test_in->m, test_in->n, d_A, lda, &lwork);

    CUDA_CHECK(cudaMalloc((void**)&d_workspace,  sizeof(data_type)*lwork));


    CUSOLVER_CHECK(cusolverDnDgetrf(handle, test_in->m, test_in->n, d_A, lda, d_workspace, d_devipiv, d_devinfo));


    CUDA_CHECK(cudaMemcpyAsync(test_in->h_A.data(), d_A, malloc_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(test_in->workspace.data(), d_workspace, malloc_size, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaMemcpyAsync(test_in->devipiv.data(), d_devipiv, min(test_in->m,test_in->n)*sizeof(int), cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaMemcpyAsync(&(test_in->devinfo), d_devinfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_devinfo));
    CUDA_CHECK(cudaFree(d_devipiv));

    CUSOLVER_CHECK(cusolverDnDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

}

void generate_A(csp_resp *test_in){
    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_int_distribution<> distrib(1,100);

    for (int i=0; i<(test_in->m * test_in->n); i++){
        int randomNumber = distrib(gen);
        test_in->h_A[i] = (data_type)randomNumber;   
    }
}

int main () {
    csp_resp test_in(10,10);
    generate_A(&test_in);
    cusolver_problem(&test_in);
    for (auto i : test_in.h_A) {
        printf("%f ", i);
    }
    printf("\n");
    for (auto i : test_in.workspace) {
        printf("%f ", i);
    }
    printf("\n");
    for (auto i : test_in.devipiv) {
        printf("%d ", i);
    }
}