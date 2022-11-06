#include "passes.h"

#include "log.h"
#include "portability.h"

#include "../rewrite.h"
#include "../type.h"

#include "../transform/ir_gen_helpers.h"

#include "list.h"
#include "dict.h"

#include <assert.h>
#include <string.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    bool disable_lowering;
    struct Dict* assigned_fn_ptrs;
    FnPtr next_fn_ptr;

    Node* god_fn;
} Context;

static const Node* process(Context* ctx, const Node* old);

static const Node* fn_ptr_as_value(IrArena* arena, FnPtr ptr) {
    return uint32_literal(arena, ptr);
}

static const Node* lower_fn_addr(Context* ctx, const Node* the_function) {
    assert(the_function->tag == Function_TAG);

    FnPtr* found = find_value_dict(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function);
    if (found) return fn_ptr_as_value(ctx->rewriter.dst_arena, *found);

    FnPtr ptr = ctx->next_fn_ptr++;
    bool r = insert_dict_and_get_result(const Node*, FnPtr, ctx->assigned_fn_ptrs, the_function, ptr);
    assert(r);
    return fn_ptr_as_value(ctx->rewriter.dst_arena, ptr);
}

/// Turn a function into a top-level entry point, calling into the top dispatch function.
static void lift_entry_point(Context* ctx, const Node* old, const Node* fun) {
    assert(old->tag == Function_TAG && fun->tag == Function_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    // For the lifted entry point, we keep _all_ annotations
    Node* new_entry_pt = function(ctx->rewriter.dst_module, old->payload.fun.params, old->payload.fun.name, rewrite_nodes(&ctx->rewriter, old->payload.fun.annotations), nodes(dst_arena, 0, NULL));

    BodyBuilder* builder = begin_body(ctx->rewriter.dst_module);

    // Put a special zero marker at the bottom of the stack so the program ends after the entry point is done
    gen_push_value_stack(builder, gen_primop_ce(builder, subgroup_active_mask_op, 0, NULL));
    gen_push_value_stack(builder, int32_literal(dst_arena, 0));

    for (size_t i = fun->payload.fun.params.count - 1; i < fun->payload.fun.params.count; i--) {
        gen_push_value_stack(builder, fun->payload.fun.params.nodes[i]);
    }

    gen_store(builder, access_decl(&ctx->rewriter, ctx->rewriter.src_module, "next_fn"), lower_fn_addr(ctx, fun));
    const Node* entry_mask = gen_primop_ce(builder, subgroup_active_mask_op, 0, NULL);
    gen_store(builder, access_decl(&ctx->rewriter, ctx->rewriter.src_module, "next_mask"), entry_mask);

    bind_instruction(builder, call_instr(dst_arena, (Call) {
        .callee = fn_addr(dst_arena, (FnAddr) { .fn = ctx->god_fn }),
        .args = nodes(dst_arena, 0, NULL)
    }));

    new_entry_pt->payload.fun.body = finish_body(builder, fn_ret(dst_arena, (Return) {
        .fn = NULL,
        .args = nodes(dst_arena, 0, NULL)
    }));
}

static const Node* process(Context* ctx, const Node* old) {
    const Node* found = search_processed(&ctx->rewriter, old);
    if (found) return found;

    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case Function_TAG: {
            Context ctx2 = *ctx;

            ctx2.disable_lowering = lookup_annotation(old, "Leaf");
            if (ctx2.disable_lowering) {
                Node* fun = recreate_decl_header_identity(&ctx->rewriter, old);
                fun->payload.fun.body = process(&ctx2, old->payload.fun.body);
                return fun;
            }

            const Node* entry_point_annotation = NULL;
            Nodes old_annotations = old->payload.fun.annotations;
            LARRAY(const Node*, new_annotations, old_annotations.count);
            size_t new_annotations_count = 0;
            for (size_t i = 0; i < old_annotations.count; i++) {
                const Node* annotation = rewrite_node(&ctx->rewriter, old_annotations.nodes[i]);
                // Entry point annotations are removed
                if (strcmp(annotation->payload.annotation.name, "EntryPoint") == 0) {
                    assert(!entry_point_annotation && "Only one entry point annotation is permitted.");
                    entry_point_annotation = annotation;
                    continue;
                }
                new_annotations[new_annotations_count] = annotation;
                new_annotations_count++;
            }

            String new_name = format_string(dst_arena, "%s_indirect", old->payload.fun.name);

            Node* fun = function(ctx->rewriter.dst_module, nodes(dst_arena, 0, NULL), new_name, nodes(dst_arena, new_annotations_count, new_annotations), nodes(dst_arena, 0, NULL));
            // register_processed(&ctx->rewriter, old, fun);

            if (entry_point_annotation)
                lift_entry_point(ctx, old, fun);

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            // Params become stack pops !
            for (size_t i = 0; i < old->payload.fun.params.count; i++) {
                const Node* old_param = old->payload.fun.params.nodes[i];
                const Node* popped = gen_pop_value_stack(bb, rewrite_node(&ctx->rewriter, extract_operand_type(old_param->type)));
                // register_processed(&ctx->rewriter, old_param, popped);
            }
            fun->payload.fun.body = finish_body(bb, rewrite_node(&ctx2.rewriter, old->payload.fun.body));
            return fun;
        }
        case FnAddr_TAG: return lower_fn_addr(ctx, old->payload.fn_addr.fn);
        case JoinPointType_TAG: return type_decl_ref(dst_arena, (TypeDeclRef) {
            .decl = find_or_process_decl(&ctx->rewriter, ctx->rewriter.src_module, "JoinPoint"),
        });
        case Let_TAG: {
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);

            const Node* old_instruction = old->payload.let.instruction;
            if (old_instruction->tag == Control_TAG) {
                BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
                /*const Node* target = rewrite_node(&ctx->rewriter, old->payload.let.tail);

                const Node* jp = bind_instruction(bb, call_instr(dst_arena, (Call) {
                    .is_indirect = false,
                    .callee = find_or_process_decl(&ctx->rewriter, ctx->src_program, "builtin_fork"),
                    .args = nodes(dst_arena, 1, (const Node*[]) { target })
                })).nodes[0];
                gen_push_value_stack(bb, jp);*/
                return finish_body(bb, fn_ret(dst_arena, (Return) { .fn = NULL, .args = nodes(dst_arena, 0, NULL) }));
            }

            return recreate_node_identity(&ctx->rewriter, old);
        }
        case TailCall_TAG: {
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);
            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            gen_push_values_stack(bb, rewrite_nodes(&ctx->rewriter, old->payload.tail_call.args));
            /*const Node* target = rewrite_node(&ctx->rewriter, old->payload.tail_call.target);

            const Node* call = call_instr(dst_arena, (Call) {
                .is_indirect = false,
                .callee = find_or_process_decl(&ctx->rewriter, ctx->src_program, "builtin_fork"),
                .args = nodes(dst_arena, 1, (const Node*[]) { target })
            });
            bind_instruction(bb, call);*/
            return finish_body(bb, fn_ret(dst_arena, (Return) { .fn = NULL, .args = nodes(dst_arena, 0, NULL) }));
        }
        case Join_TAG: {
            //if (ctx->disable_lowering)
            //    return recreate_node_identity(&ctx->rewriter, old);

            BodyBuilder* bb = begin_body(ctx->rewriter.dst_module);
            gen_push_values_stack(bb, rewrite_nodes(&ctx->rewriter, old->payload.join.args));

            /*const Node* jp = rewrite_node(&ctx->rewriter, old->payload.join.join_point);

            const Node* call = call_instr(dst_arena, (Call) {
                .is_indirect = false,
                .callee = find_or_process_decl(&ctx->rewriter, ctx->src_program, "builtin_join"),
                .args = nodes(dst_arena, 1, (const Node*[]) { jp })
            });
            bind_instruction(bb, call);*/
            return finish_body(bb, fn_ret(dst_arena, (Return) { .fn = NULL, .args = nodes(dst_arena, 0, NULL) }));
        }
        case Call_TAG: {
            // allow function addresses to survive as part of direct calls!
            const Node* callee = old->payload.call_instr.callee;
            if (callee->tag == FnAddr_TAG) {
                const Node* fn = callee->payload.fn_addr.fn;
                assert(lookup_annotation(fn, "Leaf"));
                return call_instr(dst_arena, (Call) {
                    .callee = fn_addr(dst_arena, (FnAddr) { .fn = rewrite_node(&ctx->rewriter, fn) }),
                    .args = rewrite_nodes(&ctx->rewriter, old->payload.call_instr.args)
                });
            } else error("Only leaf calls should survive to this stage!");
        }
        case PtrType_TAG: {
            const Node* pointee = old->payload.ptr_type.pointed_type;
            if (pointee->tag == FnType_TAG) {
                const Type* emulated_fn_ptr_type = int32_type(ctx->rewriter.dst_arena);
                return emulated_fn_ptr_type;
            }
            SHADY_FALLTHROUGH
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

void generate_top_level_dispatch_fn(Context* ctx) {
    assert(ctx->god_fn->tag == Function_TAG);
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    BodyBuilder* loop_body_builder = begin_body(ctx->rewriter.dst_module);

    const Node* next_function = gen_load(loop_body_builder, access_decl(&ctx->rewriter, ctx->rewriter.src_module, "next_fn"));

    struct List* literals = new_list(const Node*);
    struct List* cases = new_list(const Node*);

    const Node* zero_lit = int32_literal(dst_arena, 0);
    Node* zero_case = lambda(ctx->rewriter.dst_module, nodes(dst_arena, 0, NULL));
    zero_case->payload.anon_lam.body = finish_body(begin_body(ctx->rewriter.dst_module), merge_break(dst_arena, (MergeBreak) {
        .args = nodes(dst_arena, 0, NULL),
    }));

    append_list(const Node*, literals, zero_lit);
    append_list(const Node*, cases, zero_case);

    Nodes old_decls = get_module_declarations(ctx->rewriter.src_module);
    for (size_t i = 0; i < old_decls.count; i++) {
        const Node* decl = old_decls.nodes[i];
        if (decl->tag == Function_TAG) {
            if (lookup_annotation(decl, "Leaf"))
                continue;

            const Node* fn_lit = lower_fn_addr(ctx, find_processed(&ctx->rewriter, decl));

            BodyBuilder* case_builder = begin_body(ctx->rewriter.dst_module);

            // TODO wrap in if(mask)
            bind_instruction(case_builder, call_instr(dst_arena, (Call) {
                .callee = fn_addr(dst_arena, (FnAddr) { .fn = find_processed(&ctx->rewriter, decl) }),
                .args = nodes(dst_arena, 0, NULL)
            }));

            Node* fn_case = lambda(ctx->rewriter.dst_module, nodes(dst_arena, 0, NULL));
            fn_case->payload.anon_lam.body = finish_body(case_builder, merge_continue(dst_arena, (MergeContinue) {
                .args = nodes(dst_arena, 0, NULL),
            }));

            append_list(const Node*, literals, fn_lit);
            append_list(const Node*, cases, fn_case);
        }
    }

    Node* default_case = lambda(ctx->rewriter.dst_module, nodes(dst_arena, 0, NULL));
    default_case->payload.anon_lam.body = unreachable(dst_arena);

    bind_instruction(loop_body_builder, match_instr(dst_arena, (Match) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .inspect = next_function,
        .literals = nodes(dst_arena, entries_count_list(literals), read_list(const Node*, literals)),
        .cases = nodes(dst_arena, entries_count_list(cases), read_list(const Node*, cases)),
        .default_case = default_case,
    }));

    destroy_list(literals);
    destroy_list(cases);

    Node* loop_inside = lambda(ctx->rewriter.dst_module, nodes(dst_arena, 0, NULL));
    loop_inside->payload.anon_lam.body = finish_body(loop_body_builder, unreachable(dst_arena));

    const Node* the_loop = loop_instr(dst_arena, (Loop) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .initial_args = nodes(dst_arena, 0, NULL),
        .body = loop_inside
    });

    BodyBuilder* dispatcher_body_builder = begin_body(ctx->rewriter.dst_module);
    bind_instruction(dispatcher_body_builder, the_loop);

    ctx->god_fn->payload.fun.body = finish_body(dispatcher_body_builder, fn_ret(dst_arena, (Return) {
        .args = nodes(dst_arena, 0, NULL),
        .fn = ctx->god_fn,
    }));
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void lower_tailcalls(SHADY_UNUSED CompilerConfig* config, Module* src, Module* dst) {
    struct Dict* ptrs = new_dict(const Node*, FnPtr, (HashFn) hash_node, (CmpFn) compare_node);
    IrArena* dst_arena = get_module_arena(dst);

    Nodes top_dispatcher_annotations = nodes(dst_arena, 0, NULL);
    Node* dispatcher_fn = function(dst, nodes(dst_arena, 0, NULL), "top_dispatcher", top_dispatcher_annotations, nodes(dst_arena, 0, NULL));

    Context ctx = {
        .rewriter = create_rewriter(src, dst, (RewriteFn) process),
        .disable_lowering = false,
        .assigned_fn_ptrs = ptrs,
        .next_fn_ptr = 1,

        .god_fn = dispatcher_fn,
    };

    rewrite_module(&ctx.rewriter);
    generate_top_level_dispatch_fn(&ctx);
    destroy_dict(ptrs);
    destroy_rewriter(&ctx.rewriter);
}
