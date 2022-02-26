# Shady

`shady` is a small shading language and IR for demonstration purposes. It is meant to map easily to SPIR-V shaders.

## Syntax

The idea is to keep the syntax C-ish (but saner).

```
struct Material {
  float3 -> float4 evaluate;
};

const Material[] materials;
extern float3 sunDir;

flat in int matId;
     in float3 normal;

void main() {
  Material mat = materials[matId];
  glFragColor = mat->evaluate(normal);
}

float4 diffuse(float3 n) {
  return float4(1) * dot(n, sunDir);
}
```

## Codegen for a divergent call

```
struct Material {
  cont (varying float3, cont (varying float4)) evaluate;
};

const uniform Material[] materials;
extern uniform float3 sunDir;

flat in varying int matId;
     in varying float3 normal;

[uniform private int _next = 0;
uniform varying int[16] _args; // parameters to forward to the next basic block

void main() {
  while (true) {
    switch (_next) {
      case 0: main_bb0(); break;
      case 1: main_bb1(args[0], args[1] as float); break;
    }
  }
}

void main_bb0() {
  varying Material mat = materials[matId];
  
  // lowered non-uniform indirect call into a loop
  main_divergent_call_loop_head(mat->evaluate, undef, true)
}

// non-uniform iteration continue
@masked
cont main_divergent_call_loop_head(varying cont (varying float4) divergent_target, varying float4 join_value, varying bool mask) {

  cont main_divergent_call_loop_exit() {
      main_end(join_value)
  }
  
  cont main_divergent_call_loop_continue() {
    uniform cont (varying float4) voted_next = first_active_lane(mask, divergent_target);
      
    // call the function uniformingly, but with a mask, and special handling of the result
    varying float4 return_join_value = callc voted_next(normal, return_wrap, nmask);
      
    // depending on if we actually executed what we just returned from, we accept the new join value or keep the old one
    varying float4 merged_join_value = mask ? return_join_value : join_value;
      
    // kill invocations which are now satisfied
    varying bool nmask = mask & voted_next != divergent_target;
        
    main_divergent_call_loop_head(divergent_target, merged_join_value, nmask)
  }

  uniform cont (varying cont (varying float4), varying float4) next = 
    any(mask) ?
      main_divergent_call_loop_continue : main_divergent_call_loop_exit;
  next()
}

void main_end(float4 a0) {
  glFragColor = a0;
  terminate()
}

@masked @function @indirect_callable
void diffuse(float3 n, cont float4 ret, varying bool mask) {
  ret(float4(1) * dot(n, sunDir), mask)
}
```

## Details of callc implementation

Lower callc by splitting the continuation in two, and saving the variables that outlive the call in the stack.

TODO: clarify when using some static storage is possible (it has something to do with recursion...)

```
@masked
cont main_divergent_call_loop_head(varying cont float4 divergent_target, varying float4 join_value, varying bool mask) {
  cont main_divergent_call_loop_head() {
    // save this state, we need it later when returning from the call
    push divergent_target;
    push join_value;
    push mask;
  
    uniform cont float4 voted_next = alive_vote(divergent_target);
    
    // call the function uniformingly, but with a mask, and special handling of the result
    voted_next(normal, return_wrap, nmask)
  }
  
  cont main_divergent_call_loop_exit() { ... }
  
  @masked @indirect_reachable
  void return_wrap(varying float4 ret_value, varying bool ret_mask) {
    // we return from the call. We now need the old state back
    varying bool mask = pop;
    varying float4 join_value = pop;
    varying divergent_target = pop;
    
    // depending on if we actually executed what we just returned from, we accept the new join value or keep the old one
    varying float4 merged_join_value = ret_mask ? ret_value : join_value;
    
    // kill invocations which are now satisfied
    varying bool nmask = mask & !ret_mask;
      
    main_divergent_call_loop_head(divergent_target, merged_join_value, nmask)
  }
  
  ...
}
```

Continuation calling conv:
 * regular parameters
 * [return address] if function
 * [execution mask] if potentially divergent

Join points 

## Varying-inference

| Case | Inferred as |
|--------------|---------|
| `extern`     | Uniform |
| `const`      | Uniform |
| `in`         | Varying |
| `out`        | Varying |
| fn parameter | Varying |