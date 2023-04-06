#include "passes.h"

#include "portability.h"

#include "../rewrite.h"

typedef struct {
    Rewriter rewriter;
} Context;

void import(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) recreate_node_identity),
    };

    rewrite_module(&ctx.rewriter);
    destroy_rewriter(&ctx.rewriter);
}
