add_library(cuda_mod OBJECT context.c device.c hook.c event.c hook.c memory.c stream.c graph.c)
# Strip -fvisibility=hidden for the wrapper TUs. The final library still uses
# a linker version script (softmig.ver) to restrict exports to cu*/nvml*, so
# internal helpers remain private while the hook symbols get DEFAULT visibility
# and correctly interpose via LD_PRELOAD for apps that link -lcuda directly.
set(CUDA_MOD_FLAGS ${LIBRARY_COMPILE_FLAGS})
list(REMOVE_ITEM CUDA_MOD_FLAGS "-fvisibility=hidden")
target_compile_options(cuda_mod PUBLIC ${CUDA_MOD_FLAGS})
target_link_libraries(cuda_mod PUBLIC nvidia-ml -lcuda)
