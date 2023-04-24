# Shady

![](logo.png)

`shady` is a small intermediate shading language and compiler for research purposes. It strives to be a testbed for improved GPU programming models, and also provide support for emulating features either missing from SPIR-V, or suffering from poor support.

Shady is ideal for projects that aim to target SPIR-V, but already make use of features that are not found in vanilla Vulkan compute. See feature support below.

Shady is used as part of the [AnyDSL](https://anydsl.github.io) to provide experimental Vulkan accelerator support. Technical discussion about shady and SPIR-V in general can be had on [this discord server](https://twitter.com/gobrosse/status/1441323225128968197)

## Feature support

Not all supported features are listed, these are just the more notable ones that are either already working, or are planned.

 * [x] True function calls (thanks to a CPS transformation)
   * Function pointers/indirect calls
   * Recursion (with a stack)
   * Divergent calls
 * [x] Arbitrary control flow inside functions: `goto`, including non-uniform is allowed
   * This makes `shady` easy to target from existing compilers.
   * Reconvergence is explicit and dataflow driven, not reliant on a CFG analysis.
 * [x] Subgroup memory (known as `simdgroup` in Metal)
 * [x] Physical pointers (cast-able pointers where the layout of objects in memory is observable)
   * [x] For 'private' memory
   * [x] For 'shared' memory
 * 'Wide' subgroup operations (with arbitrary types)
   * [x] Ballot
   * [ ] Shuffles
 * [ ] Int8, Int16 and Int64 support everywhere
 * [ ] FP 64 emulation
 * [x] Generic (tagged) pointers
 * [x] Printf debug support
 * [x] Adapt code generation to the target through a `runtime` component.

## Platform support

 * Compiles on Windows/MacOS/Linux with any C11 compliant toolchain: GCC, Clang and recent versions of MSVC
   * Windows SDKs older than 10.0.20348.0 are missing important C11 features and are unsupported.
   * Will run as far back as [Windows XP](https://mastodon.gamedev.place/@gob/109580697549344123) - using MinGW based toolchains.
   * We ran the compiler on IA32, AMD64, AArch64 and RISCV 64 machines with no issues.
 * The following Vulkan drivers have been tested:
   * [x] `radv` Open-source mesa driver for AMD GPUs
     * Tested on multiple RDNA2 and GCN devices
   * [x] `anv` Open-source mesa driver for Intel GPUs
   * [x] NVidia proprietary drivers (requiring a [small hack](https://github.com/Hugobros3/shady/commit/f3ef83dbff7f29654fc11f8901ba67494864c085))
   * [x] Intel proprietary Windows drivers for Intel HD cards
     * [ ] Xe (Arc) cards come with their own driver, which can't run currently due to missing `Int64` support
   * [ ] MoltenVK does not work due to buggy Metal drivers currently miscompiling SIMD intrinsics.
     * Might work on non-apple sillicon devices
   * [ ] Imagination closed-source driver on the VisionFive 2 board: driver crash

Additionally, the compiler supports alternative backends:
 * GLSL (untested - no runtime component yet)
 * ISPC (no runtime component either, but useful for debugging on the host)

Metal shading language and C backends are on the table in the future.

## Compiler design

* Semi-immutable impure IR:
    * Qualified value types (`uniform` or `varying`), type system can enforce uniformity for sensitive operations
    * Meta-instructions for conventional structured control flow (`if`, `match`, `loop`), no convergence annotations required
    * Experimental new dynamically structured control flow primitives (paper/writeup coming later)
* Nodes are either nominal (top-level declarations, variables and basic blocks) or structural (everything else). Structural nodes are immutable and subject to hash-consing and folding ops during construction.
* Shady is written in standard C11 with extensive use of x-macros to define the grammar, operations etc, and generate much of the boilerplate code (node hashing, rewriting, visitors, ...)

## Language syntax

The syntax is under construction. See [grammar.md](grammar.md) for a hopefully not-that-outdated grammar file.

Initially the idea was to have C-like syntax, but that proved annoying so the only significant remnant is the type-before-id aesthetic.

The current syntax reflects the IR quite closely, and is not meant to be easy to write real programs in directly. In the future we might add enough syntactic sugar to make that feasible though.

```
// line comments are supported
fn identity varying int(varying i32 i) {
    return (i);
};

fn f i32(varying i32 i) {
    val j = call (identity, i);
    val k = add (j, 1);
    return (k);
};

const i32 answer = 42;
```

The textual syntax allows nesting basic blocks inside functions. The syntax is superficially similar to C labels, but with an added parameters list. Note that this is mostly for making handwritten examples look nicer, the actual nesting of functions/continuations is determined by the CFG analysis after name binding.

```
fn f i32(varying bool b) {
    jump bb1(7);

    cont bb1(varying i32 n) {
        branch (b, bb2(), bb3(n));
    }

    cont bb2() {
        return (0);
    }

    cont bb3(varying i32 n) {
        return (n);
    }
};
```
