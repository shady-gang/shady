# Shady

`shady` is a small shading language and IR for research purposes.

Unlike conventional shading languages, `shady` internally represent control flow using continuations and has annotations for varying and uniform values.
The plan is to develop a alternative programming model to structured control flow, focusing instead of keeping control flow uniform and using masks to handle divergence explicitly.

Technical discussion for shady and SPIR-V in general can be had on [this discord server](https://twitter.com/gobrosse/status/1441323225128968197)

## Design

 * Written in C11
 * Hash-consed IR with nominal (functions) and structural (everything else) nodes. Structural nodes are immutable.
 * The function node encodes both functions and continuations. For many intents, the distinction does not matter.
 * Function bodies (named "blocks" inside the IR) are made out of lists of instructions without control flow, followed by a terminator
 * Implicit scoping like in Thorin, but using the CFG to establish scopes rather than the uses of params

## Goals

 * Achieve code generation for arbitrarily complex/divergent/indirect programs using magic (hacks)
 * Emulate missing features by polyfill/shims
 * SPIR-V is currently the only target, but since other shading languages offer similar programming models, this could be extended to GLSL, HLSL, MSL, WGSL, ...

## Syntax

The syntax is under construction. See [grammar.md](GRAMMAR.MD) for a hopefully not-that-outdated grammar file.

Initially the idea was to have C-like syntax, but that was annoying so the only significant remnant is the type-before-id convention.

The current syntax reflects the IR quite closely, and is not meant to be easy to write real programs in directly. In the future we might add enough syntactic sugar to make that feasible though.

```
// line comments are supported
fn identity = varying int(varying int i) {
    return i;
};

fn f = int (varying int i) {
    let j = call identity i;
    let k = add j 1;
    return k;
};

const answer = 42;
```

The textual syntax allows nesting functions. The syntax is superficially similar to C labels, but with an added parameters list. 
Note that this is mostly for making handwritten examples look nicer, the actual nesting of functions/continuations is determined by the CFG analysis after name binding.

```
fn f = int (varying bool b) {
    jump bb1 7;

    bb1: (varying int n) {
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