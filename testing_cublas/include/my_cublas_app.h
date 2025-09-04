#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdio>
#include "em_mm_types.h"
#include "common.h"

using data_type= double;

size_response test_size_shenans(
    const std::vector<data_type> A, 
    const std::vector<data_type> B, 
    const std::vector<data_type> C);

MMStatus matrix_multiplication(
    std::vector<data_type>& A, 
    std::vector<data_type>& B, 
    std::vector<data_type>& C
    );