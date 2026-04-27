#include "include/libcuda_hook.h"

CUresult cuStreamCreate(CUstream *phstream, unsigned int flags){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamCreate,phstream,flags);
    return res;
}

CUresult cuStreamDestroy_v2 ( CUstream hStream ){
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamDestroy_v2,hStream);
}

CUresult cuStreamSynchronize(CUstream hstream){
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamSynchronize,hstream);
    return res;
}
