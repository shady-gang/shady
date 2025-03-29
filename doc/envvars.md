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

## Dumping

 * `SHADY_CFG_SCOPE_ONLY`: Do not print basic block contents when generating a CFG.

## Runner

 * `SHADY_OVERRIDE_SPV`: For the Vulkan runner, load that file instead of what the JIT pipeline produced
 * `SHADY_OVERRIDE_PTX`: For the CUDA runner, load that file instead of what the JIT pipeline produced