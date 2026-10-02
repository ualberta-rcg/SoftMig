/**
 * @file graph.c
 * @brief CUDA graph API hooks.
 *
 * Only cuGraphLaunch is intercepted (to invoke rate_limiter for SM
 * throttling). All other graph functions are pure pass-throughs with no
 * SoftMig-specific logic, so we let dlsym resolve them directly from
 * libcuda. This avoids signature/macro conflicts with CUDA 13, where several
 * graph functions are renamed to _v2 versions with different signatures.
 */
#include "include/libcuda_hook.h"

extern void rate_limiter(int grids, int blocks);
extern int pidfound;

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
	SOFTMIG_PASSIVE_FORWARD(cuGraphLaunch, hGraphExec, hStream);
	if (pidfound == 1) {
		rate_limiter(0, 0);
	}
	return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGraphLaunch,hGraphExec,hStream);
}

CUresult cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream) {
	return cuGraphLaunch(hGraphExec, SOFTMIG_PTSZ_STREAM(hStream));
}
