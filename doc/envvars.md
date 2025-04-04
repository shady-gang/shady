# Environment variables

## Logging

 * `SHADY_LOG_LEVEL=level` sets the log level, where `level` is:
   * `error`: show-stopping errors
   * `warn`: useful warnings
   * `info`: minor warnings and notices
   * `debug[v[v]]`: increasingly verbose info for debugging
 * `SHADY_NODE_DUMP=opt1,opt2,...,optN` configures node printing, where `opti` are:
   * `color=<1|0>`: Enables colored terminal output
   * `function-body=<1|0>`: whether to print function bodies (not doing so is useful when looking for globals)
   * `scheduled=<1|0>`: Whether to schedule the nodes when printing. Disabling scheduling can help diagnosing a broken IR.
   * `internal=<1|0>`: Print 'internal' nodes
   * `generated=<1|0>`: Print 'generated' nodes
 * `SHADY_SUPER_VERBOSE_NODE_DEBUG`: Prints the ID of every node everywhere
 * `SHADY_DUMP_CLEAN_ROUNDS`: Prints the module for each optimization loop iteration
 * `SHADY_LOG_PASS=<1|pass_name1,pass_name2,...>`: Logs the module after each pass or the pass names provided

## DebugPrintf tracing

 * `SHADY_PRINTF_TRACE=...`
   * `stack-size`: prints the stack size everytime it changes
   * `stack-access`: prints on stack access (push/pop)
   * `max-stack-size`: prints the max stack size at the end of the program
   * `memory-access`: prints on every emulated memory access
   * `top-function`: prints on every iteration of the top function
   * `scratch-base-addr`: prints the computed base address for the subgroup in scratch

## Dumping

 * `SHADY_CFG_SCOPE_ONLY`: Do not print basic block contents when generating a CFG.

## Vulkan Runner

 * `SHADY_OVERRIDE_SPV`: For the Vulkan runner, load that file instead of what the JIT pipeline produced

## CUDA runner

 * `SHADY_OVERRIDE_CU`: Load that file instead of what the JIT pipeline produced
 * `SHADY_OVERRIDE_PTX`: Load that file instead of what the JIT pipeline produced
 * `SHADY_NVRTC_PARAMS`: Load additional arguments for nvrtc from that file 
 * `SHADY_CUDA_RUNNER_DUMP`: Dumps the .cu, .ptx and .cubin files generated during compilation