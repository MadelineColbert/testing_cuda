#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <cstdio>

void cusolver_problem() {
    cusolverDnHandle_t handle;
    cudaStream_t stream;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnCreate(&handle);
    cusolverDnSetStream(handle, stream);

    double* A;
    double* workspace;
    int* devipiv;
    int* devinfo;


    int m=0;
    int n=0;
    int lda=0;

    cusolverDnDgetrf(handle, m,n, A, lda, workspace, devipiv, devinfo);

    cudaStreamSynchronize(stream);

    cusolverDnDestroy(handle);
    cudaStreamDestroy(stream);

}

int main () {
    printf("Testing Make");
}