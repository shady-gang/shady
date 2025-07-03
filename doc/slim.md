Shady features a small ad-hoc textual programming language called Slim (mom's spaghetti), which was used initially for bringup of the compiler.

We currently use this language as part of our testing infrastructure.

## Language syntax

The textual syntax of the language is C-like in that return types come first. Variance annotations are supported.
Overall the language is structurally close to SPIR-V and LLVM, very much on purpose.

`.slim` files parse directly into IR, but use a [custom instruction set](../src/frontend/slim/extinst.spv-shady-slim-frontend.grammar.json) to represent the syntactic sugar such as name binding, mutable variables, dereferencing, etc.
These get lowered away by [passes in the frontend](../src/frontend/slim/).

```
// line comments are supported
fn identity varying i32(varying i32 i) {
    return (i);
};

fn f i32(varying i32 i) {
    val j = call(identity, i);
    val k = add(j, 1);
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
