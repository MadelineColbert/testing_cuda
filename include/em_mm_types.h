#pragma once

typedef enum {
    MMSUCCESS=0,
    SIZE_MISMATCH=1
} MMStatus;

typedef struct size_response {
    int sz;
    MMStatus error_code;
} size_response;