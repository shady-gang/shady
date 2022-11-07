# Shady

`shady` is a small shading language and IR for research purposes. It strives to be a testbed for GPU programming models, and also provide support for emulating features either missing from SPIR-V, or suffering from poor support.

Technical discussion about shady and SPIR-V in general can be had on [this discord server](https://twitter.com/gobrosse/status/1441323225128968197)

## Design

 * Written in standard C11 with extensive use of x-macros to define the grammar, operations etc, and generate much of the boilerplate.
 * Nodes are either nominal (top-level declarations, variables and basic blocks) or structural (everything else). Structural nodes are immutable and subject to hash-consing and folding ops during construction.
 * All values are qualified `uniform` or `varying` and typing rules care about that
 * Statically structured control flow constructs (selection and iteration constructs) can be represented by special instructions
 * Experimental new dynamically structured control flow primitives (paper/writeup coming later)

## Goals

 * Achieve code generation for arbitrarily complex/divergent/indirect programs using magic (aka a big pile of hacks)
 * Emulate missing features where support is missing, while using extensions opportunistically
 * SPIR-V is currently the primary target, but since other shading languages offer similar programming models, this can be extended to GLSL, HLSL, MSL, WGSL, ...

## Syntax

The syntax is under construction. See [grammar.md](grammar.md) for a hopefully not-that-outdated grammar file.

Initially the idea was to have C-like syntax, but that proved annoying so the only significant remnant is the type-before-id aesthetic.

The current syntax reflects the IR quite closely, and is not meant to be easy to write real programs in directly. In the future we might add enough syntactic sugar to make that feasible though.

```
// line comments are supported
fn identity varying int(varying int i) {
    return i;
};

fn f i32(varying i32 i) {
    let j = call identity i;
    let k = add j 1;
    return k;
};

const i32 answer = 42;
```

The textual syntax allows nesting basic blocks inside functions. The syntax is superficially similar to C labels, but with an added parameters list. Note that this is mostly for making handwritten examples look nicer, the actual nesting of functions/continuations is determined by the CFG analysis after name binding.

```
fn f i32(varying bool b) {
    jump bb1 7;

    bb1: (varying i32 n) {
        return n;
    }
};
```

This is the current syntax for externals/global/IO variables. The `extern` variables are mapped to push constants/ubo/ssbos at the runtime's discretion.

```
input   int x;
output  int y;
shared  int z;
private int w;
extern  int a;
```