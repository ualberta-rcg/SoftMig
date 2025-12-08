#!/bin/bash
# CUDA MEMORY ALLOCATION TEST SUITE (Compute Canada)
# Tests: cudaMalloc, cudaMallocManaged, cudaMallocAsync
#        cuMemAlloc, cuMemAllocManaged, cuMemCreate/VMM

export LIBCUDA_LOG_LEVEL=5
module load cuda/12.2

###############################################
# DIAGNOSTIC FUNCTIONS
###############################################

check_softmig_setup() {
    echo "=== SoftMig Setup Check ==="
    
    # Check LD_PRELOAD
    if [ -z "$LD_PRELOAD" ]; then
        echo "  ⚠️  LD_PRELOAD is not set"
        echo "     Library may be preloaded via /etc/ld.so.preload"
        if [ -f /etc/ld.so.preload ]; then
            if grep -q "libsoftmig.so" /etc/ld.so.preload 2>/dev/null; then
                LIB_PATH=$(grep "libsoftmig.so" /etc/ld.so.preload | head -1)
                echo "  ✅ Found libsoftmig.so in /etc/ld.so.preload: $LIB_PATH"
                if [ -f "$LIB_PATH" ]; then
                    echo "     Library file exists and is readable"
                else
                    echo "     ⚠️  Library file NOT FOUND at: $LIB_PATH"
                fi
            else
                echo "  ⚠️  libsoftmig.so not found in /etc/ld.so.preload"
            fi
        fi
    else
        echo "  ✅ LD_PRELOAD is set: $LD_PRELOAD"
        if [ ! -f "$LD_PRELOAD" ]; then
            echo "  ⚠️  Warning: Library file not found at: $LD_PRELOAD"
        fi
    fi
    
    # Check config files (softmig uses config files, not env vars)
    CONFIG_DIR="/var/run/softmig"
    if [ -d "$CONFIG_DIR" ]; then
        CONFIG_FILES=$(ls $CONFIG_DIR/*.conf 2>/dev/null)
        if [ -n "$CONFIG_FILES" ]; then
            echo "  ✅ Found config files:"
            for conf in $CONFIG_FILES; do
                echo "     $conf"
                echo "     Contents:"
                cat "$conf" | sed 's/^/       /'
            done
        else
            echo "  ⚠️  No config files found in $CONFIG_DIR"
            echo "     SoftMig will be in passive mode (no limits enforced)"
        fi
    else
        echo "  ⚠️  Config directory does not exist: $CONFIG_DIR"
    fi
    
    # Check log directory permissions
    LOG_DIR="/var/log/softmig"
    echo ""
    echo "  Checking log directory: $LOG_DIR"
    if [ -d "$LOG_DIR" ]; then
        if [ -w "$LOG_DIR" ]; then
            echo "  ✅ Log directory exists and is writable"
        else
            echo "  ⚠️  Log directory exists but is NOT writable (permissions issue)"
            ls -ld "$LOG_DIR" | sed 's/^/     /'
        fi
    else
        echo "  ⚠️  Log directory does not exist"
        echo "     Attempting to create it (may fail without root)..."
        mkdir -p "$LOG_DIR" 2>/dev/null
        if [ -d "$LOG_DIR" ]; then
            echo "  ✅ Log directory created"
        else
            echo "  ❌ Failed to create log directory (need root or proper permissions)"
        fi
    fi
    
    # Test if we can create a log file
    TEST_LOG="$LOG_DIR/test_write_$$.log"
    if echo "test" > "$TEST_LOG" 2>/dev/null; then
        echo "  ✅ Can write to log directory"
        rm -f "$TEST_LOG"
    else
        echo "  ❌ Cannot write to log directory (this will prevent logging!)"
        echo "     Logs may be in /tmp/softmig_pid*.log if /var/log is not writable"
    fi
    
    echo ""
}

check_process_results() {
    echo "=== Checking Test Results ==="
    echo ""
    
    # Get PIDs from background jobs
    PIDS=$(jobs -p 2>/dev/null)
    if [ -z "$PIDS" ]; then
        echo "  ⚠️  No background jobs found. Wait a moment and run:"
        echo "     bash -c 'source test_softmig.sh; check_process_results'"
        return
    fi
    
    # Map test numbers to API types
    declare -A TEST_TYPES=(
        [1]="cudaMalloc (runtime API)"
        [2]="cudaMallocManaged (runtime API)"
        [3]="cudaMallocAsync (runtime API)"
        [4]="cuMemAlloc (driver API)"
        [5]="cuMemAllocManaged (driver API)"
        [6]="cuMemCreate/cuMemMap (VMM API)"
    )
    
    TEST_NUM=1
    for PID in $PIDS; do
        TEST_TYPE="${TEST_TYPES[$TEST_NUM]}"
        echo "--- Test $TEST_NUM: $TEST_TYPE (PID $PID) ---"
        
        # Check if process is running
        if ps -p $PID > /dev/null 2>&1; then
            echo "  ✅ Process is RUNNING"
            if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "^$PID$"; then
                MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "^$PID," | cut -d',' -f2 | tr -d ' ')
                echo "  ✅ In nvidia-smi (using $MEM)"
            else
                echo "  ⚠️  Running but NOT in nvidia-smi (allocation may have failed)"
            fi
        else
            echo "  ❌ Process NOT RUNNING (may have exited/failed)"
        fi
        echo ""
        TEST_NUM=$((TEST_NUM + 1))
    done
    
    echo "=== nvidia-smi Summary ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
    echo ""
}

###############################################
# MAIN TEST SUITE
###############################################

# 1) cudaMalloc (runtime API)
(
python3 - << 'EOF'
import ctypes, os, time
cuda = ctypes.CDLL(os.environ["CUDA_HOME"] + "/lib64/libcudart.so")
p = ctypes.c_void_p()
cuda.cudaMalloc(ctypes.byref(p), 1024*1024*1024)
print("cudaMalloc: allocated 1GB, PID =", os.getpid())
while True: time.sleep(10)
EOF
) &
sleep 1
echo "=== After Test 1: cudaMalloc ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
echo ""


# 2) cudaMallocManaged (runtime UM)
(
python3 - << 'EOF'
import ctypes, os, time
cuda = ctypes.CDLL(os.environ["CUDA_HOME"] + "/lib64/libcudart.so")
p = ctypes.c_void_p()
cuda.cudaMallocManaged(ctypes.byref(p), 1024*1024*1024, 1)
print("cudaMallocManaged: allocated 1GB, PID =", os.getpid())
while True: time.sleep(10)
EOF
) &
sleep 1
echo "=== After Test 2: cudaMallocManaged ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
echo ""


# 3) cudaMallocAsync (runtime async allocator)
(
python3 - << 'EOF'
import ctypes, os, time
cuda = ctypes.CDLL(os.environ["CUDA_HOME"] + "/lib64/libcudart.so")

# create stream
stream = ctypes.c_void_p()
cuda.cudaStreamCreate(ctypes.byref(stream))

p = ctypes.c_void_p()
cuda.cudaMallocAsync(ctypes.byref(p), 1024*1024*1024, stream)
print("cudaMallocAsync: allocated 1GB, PID =", os.getpid())
while True: time.sleep(10)
EOF
) &
sleep 1
echo "=== After Test 3: cudaMallocAsync ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
echo ""


# 4) cuMemAlloc (driver API v1)
(
python3 - << 'EOF'
import ctypes, os, time
cu = ctypes.CDLL("libcuda.so")
cu.cuInit(0)
p = ctypes.c_void_p()
cu.cuMemAlloc(ctypes.byref(p), 1024*1024*1024)
print("cuMemAlloc: allocated 1GB, PID =", os.getpid())
while True: time.sleep(10)
EOF
) &
sleep 1
echo "=== After Test 4: cuMemAlloc ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
echo ""


# 5) cuMemAllocManaged (driver API v2 UM)
(
python3 - << 'EOF'
import ctypes, os, time
cu = ctypes.CDLL("libcuda.so")
cu.cuInit(0)
p = ctypes.c_void_p()
cu.cuMemAllocManaged(ctypes.byref(p), 1024*1024*1024, 1)
print("cuMemAllocManaged: allocated 1GB, PID =", os.getpid())
while True: time.sleep(10)
EOF
) &
sleep 1
echo "=== After Test 5: cuMemAllocManaged ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
echo ""


# 6) cuMemCreate + cuMemMap (VMM API — hardest to intercept)
(
python3 - << 'EOF'
import ctypes, os, time
cu = ctypes.CDLL("libcuda.so")
cu.cuInit(0)

size = 1024*1024*1024
handle = ctypes.c_ulonglong()
props = (ctypes.c_ulonglong*5)(0,0,0,0,0)

cu.cuMemCreate(ctypes.byref(handle), size, props, 0)

addr = ctypes.c_void_p()
cu.cuMemAddressReserve(ctypes.byref(addr), size, 0, 0, 0)
cu.cuMemMap(addr, size, 0, handle, 0)

print("cuMemCreate/cuMemMap: allocated 1GB, PID =", os.getpid())
while True: time.sleep(10)
EOF
) &
sleep 1
echo "=== After Test 6: cuMemCreate/cuMemMap ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
echo ""

###############################################
echo ""
echo "All allocation tests started."
echo ""
echo "=== Quick Commands ==="
echo "  Check results:     bash -c 'source test_softmig.sh; check_process_results'"
echo "  View nvidia-smi:   nvidia-smi"
echo "  View jobs:         jobs -l"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "  Check logs:        tail -f /var/log/softmig/${SLURM_JOB_ID}.log"
else
    echo "  Check logs:        tail -f /var/log/softmig/*.log"
fi
echo ""
echo "=== Notes ==="
echo "  ⚠️  IMPORTANT: Python's ctypes.CDLL bypasses dlsym, so hooks may NOT work!"
echo "  - SoftMig uses config files in /var/run/softmig/, not environment variables"
echo "  - If no config file exists, SoftMig runs in passive mode (no limits)"
echo "  - To test properly, use applications that link against CUDA at compile time"
echo "    (PyTorch, TensorFlow, or C programs) instead of ctypes.CDLL"
echo ""
###############################################

# Run setup check
check_softmig_setup

# Wait a moment for processes to start, then show results
sleep 2
check_process_results

# Show main log file (SLURM job ID log)
echo ""
echo "=== Main Log File (Last 20 lines) ==="
LOG_DIR="/var/log/softmig"
if [ -n "$SLURM_JOB_ID" ]; then
    MAIN_LOG="$LOG_DIR/${SLURM_JOB_ID}.log"
else
    # Find most recent log file
    MAIN_LOG=$(ls -t $LOG_DIR/*.log 2>/dev/null | head -1)
fi

if [ -n "$MAIN_LOG" ] && [ -f "$MAIN_LOG" ]; then
    echo "  Log file: $MAIN_LOG"
    echo ""
    tail -20 "$MAIN_LOG" | sed 's/^/  /'
    echo ""
    
    # Check for process summing logs
    echo "=== Checking for Process Summing Logs ==="
    if grep -q "into nvmlDeviceGetMemoryInfo" "$MAIN_LOG" 2>/dev/null; then
        COUNT=$(grep -c "into nvmlDeviceGetMemoryInfo" "$MAIN_LOG" 2>/dev/null)
        echo "  ✅ Found $COUNT calls to nvmlDeviceGetMemoryInfo"
    else
        echo "  ❌ No calls to nvmlDeviceGetMemoryInfo found"
        echo "     (nvidia-smi may not be calling the hooked function)"
    fi
    
    if grep -q "get_current_device_memory_usage.*Found.*processes" "$MAIN_LOG" 2>/dev/null; then
        echo "  ✅ Found process summing logs:"
        grep "get_current_device_memory_usage.*Found.*processes" "$MAIN_LOG" | tail -3 | sed 's/^/    /'
    else
        echo "  ❌ No process summing logs found"
        echo "     (Function may not be reaching the summing code)"
    fi
    
    if grep -q "Manual sum" "$MAIN_LOG" 2>/dev/null; then
        echo "  ✅ Found manual sum logs:"
        grep "Manual sum" "$MAIN_LOG" | tail -3 | sed 's/^/    /'
    else
        echo "  ❌ No manual sum logs found"
    fi
    
    if grep -q "Process\[.*PID=" "$MAIN_LOG" 2>/dev/null; then
        echo "  ✅ Found process detail logs (showing last 5):"
        grep "Process\[.*PID=" "$MAIN_LOG" | tail -5 | sed 's/^/    /'
    else
        echo "  ❌ No process detail logs found"
    fi
    
    echo ""
    echo "=== Debug: get_current_device_memory_usage calls ==="
    if grep -q "get_current_device_memory_usage" "$MAIN_LOG" 2>/dev/null; then
        echo "  Found calls (showing last 5):"
        grep "get_current_device_memory_usage" "$MAIN_LOG" | tail -5 | sed 's/^/    /'
    else
        echo "  ❌ No calls to get_current_device_memory_usage found"
    fi
else
    echo "  ⚠️  No main log file found"
    echo "     (Logs may be in $LOG_DIR/ or /tmp/)"
fi
echo ""
