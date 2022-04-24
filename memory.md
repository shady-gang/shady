# ramblings about memory

## typed pointers or not

 * modern shit likes to forgo pointer typing (ie LLVM15)
 * problem: SPIR-V & co by default don't like bitcasting
   * this can be emulated ...
   * but the emulating code itself has to use typed pointers
   * so the IR should retain typed pointers
   * at least for now
   * maybe the element type can be made optional

## state of the stack when entering the main loop

```
function_id
param1
param2
...
paramN
restorable_live_value1
restorable_live_value2
...
restorable_live_valueN
```