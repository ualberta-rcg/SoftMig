#!/bin/bash
# Build the real gpu-burn (wilicc) once per CUDA module into
# build-cuda<ver>/gpu-burn/{gpu_burn,compare.ptx}. Runs as ONE reservation
# job (never on the login node). Source: $GPU_BURN_SRC (default ~/gpu-burn).
#
#   test/build_gpuburn.sh [12.2 12.6 12.9 13.2]
set -u
SOFTMIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${GPU_BURN_SRC:-$HOME/gpu-burn}"
if [ $# -gt 0 ]; then VERSIONS=("$@"); else VERSIONS=(12.2 12.6 12.9 13.2); fi
[ -f "$SRC/gpu_burn-drv.cpp" ] || { echo "no gpu-burn source at $SRC" >&2; exit 2; }

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:l40s.4:1 --cpus-per-task=8 --mem=8G --time=00:20:00 \
     bash -lc "
cd '$SOFTMIG_ROOT'
for v in ${VERSIONS[*]}; do
  module purge >/dev/null 2>&1; module load StdEnv/2023 cuda/\$v >/dev/null 2>&1 || { echo \"[\$v] module load failed\"; continue; }
  out=build-cuda\$v/gpu-burn; mkdir -p \$out
  tmp=\$(mktemp -d); cp '$SRC'/gpu_burn-drv.cpp '$SRC'/compare.cu '$SRC'/Makefile \$tmp/
  # CUDA 13 made cuCtxCreate the 4-argument _v4 form
  case \$v in 13.*) sed -i 's/cuCtxCreate(&d_ctx, 0, d_dev)/cuCtxCreate(\&d_ctx, NULL, 0, d_dev)/' \$tmp/gpu_burn-drv.cpp ;; esac
  if make -C \$tmp CUDAPATH=\"\$CUDA_HOME\" COMPUTE=89 gpu_burn > \$out/build.log 2>&1; then
    cp \$tmp/gpu_burn \$tmp/compare.ptx \$out/ && echo \"[\$v] OK \$(ldd \$out/gpu_burn | grep -o 'cudacore/[0-9.]*' | head -1)\"
  else
    echo \"[\$v] BUILD FAILED (see \$out/build.log)\"; tail -3 \$out/build.log
  fi
  rm -rf \$tmp
done
"
