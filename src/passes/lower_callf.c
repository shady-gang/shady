#include "shady/ir.h"

#include "../rewrite.h"
#include "../type.h"
#include "../log.h"
#include "../local_array.h"

#include "list.h"
#include "dict.h"

#include <assert.h>

typedef uint32_t FnPtr;

typedef struct Context_ {
    Rewriter rewriter;
    struct Dict* assigned_fn_ptrs;
    FnPtr next_fn_ptr;

    const Node* god_fn;
    struct List* new_decls;
} Context;

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

static void gen_push_value_stack(IrArena* arena, Nodes* instructions, const Node* value) {
    *instructions = append_nodes(arena, *instructions, let(arena, (Let) {
        .variables = nodes(arena, 0, NULL),
        .instruction = prim_op(arena, (PrimOp) {
            .op = push_stack_op,
            .operands = nodes(arena, 2, (const Node*[]) { without_qualifier(value->type), value })
        })
    }));
}

static void gen_push_values_stack(IrArena* arena, Nodes* instructions, Nodes values) {
    for (size_t i = values.count - 1; i < values.count; i--) {
        const Node* value = values.nodes[i];
        gen_push_value_stack(arena, instructions, value);
    }
}

static void gen_push_fn_stack(IrArena* arena, Nodes* instructions, const Node* fn_ptr) {
    const Type* ret_param_type = int_type(arena);

    *instructions = append_nodes(arena, *instructions, let(arena, (Let) {
        .variables = nodes(arena, 0, NULL),
        .instruction = prim_op(arena, (PrimOp) {
            .op = push_stack_uniform_op,
            .operands = nodes(arena, 2, (const Node*[]) { ret_param_type, fn_ptr })
        })
    }));
}

static const Node* gen_pop_fn_stack(IrArena* arena, Nodes* instructions, String var_name) {
    const Type* ret_param_type = int_type(arena);
    const Type* q_ret_param_type = qualified_type(arena, (QualifiedType) {.type = ret_param_type, .is_uniform = true});

    const Node* ret_tmp_vars[] = { var(arena, q_ret_param_type, var_name)};
    *instructions = append_nodes(arena, *instructions, let(arena, (Let) {
        .variables = nodes(arena, 1, ret_tmp_vars),
        .instruction = prim_op(arena, (PrimOp) {
            .op = pop_stack_uniform_op,
            .operands = nodes(arena, 1, (const Node*[]) { ret_param_type })
        })
    }));
    return ret_tmp_vars[0];
}

static const Node* gen_pop_value_stack(IrArena* arena, Nodes* instructions, String var_name, const Type* type) {
    const Type* q_type = qualified_type(arena, (QualifiedType) {.type = type, .is_uniform = false});
    const Node* ret_tmp_vars[] = { var(arena, q_type, var_name)};
    *instructions = append_nodes(arena, *instructions, let(arena, (Let) {
        .variables = nodes(arena, 1, ret_tmp_vars),
        .instruction = prim_op(arena, (PrimOp) {
            .op = pop_stack_uniform_op,
            .operands = nodes(arena, 1, (const Node*[]) { type })
        })
    }));
    return ret_tmp_vars[0];
}

static Nodes gen_pop_values_stack(IrArena* arena, Nodes* instructions, String var_name, const Nodes types) {
    LARRAY(const Node*, tmp, types.count);
    for (size_t i = 0; i < types.count; i++) {
        tmp[i] = gen_pop_value_stack(arena, instructions, format_string(arena, "%s_%d", var_name, (int) i), types.nodes[i]);
    }
    return nodes(arena, types.count, tmp);
}

static const Node* fn_ptr_as_value(IrArena* arena, FnPtr ptr) {
    return int_literal(arena, (IntLiteral) {
        .value = ptr
    });
}

static const Node* callee_to_ptr(Context* ctx, const Node* callee) {
    const Type* ret_param_type = int_type(ctx->rewriter.dst_arena);

    if (callee->tag != Function_TAG) {
        assert(is_subtype(ret_param_type, without_qualifier(callee)));
        return callee;
    }

    FnPtr* found = find_value_dict(const Node*, FnPtr, ctx->assigned_fn_ptrs, callee);
    if (found) return fn_ptr_as_value(ctx->rewriter.dst_arena, *found);

    FnPtr ptr = ctx->next_fn_ptr++;
    bool r = insert_dict_and_get_result(const Node*, FnPtr, ctx->assigned_fn_ptrs, callee, ptr);
    assert(r);
    return fn_ptr_as_value(ctx->rewriter.dst_arena, ptr);
}

static const Node* lower_callf_process(Context* ctx, const Node* old) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;
    switch (old->tag) {
        case GlobalVariable_TAG:
        case Constant_TAG: {
            Node* new = recreate_decl_header_identity(&ctx->rewriter, old);
            recreate_decl_body_identity(&ctx->rewriter, old, new);
            return new;
        }
        case Function_TAG: {
            FnAttributes nattrs = old->payload.fn.atttributes;
            nattrs.entry_point_type = NotAnEntryPoint;
            String new_name = nattrs.is_continuation ? old->payload.fn.name : format_string(dst_arena, "%s_leaf", old->payload.fn.name);
            Node* fun = fn(dst_arena, nattrs, new_name, old->payload.fn.params, nodes(dst_arena, 0, NULL));

            if (old->payload.fn.atttributes.entry_point_type != NotAnEntryPoint) {
                Node* new_entry_pt = fn(dst_arena, old->payload.fn.atttributes, old->payload.fn.name, old->payload.fn.params, nodes(dst_arena, 0, NULL));
                append_list(const Node*, ctx->new_decls, new_entry_pt);

                Nodes instructions = nodes(dst_arena, 0, NULL);
                for (size_t i = fun->payload.fn.params.count - 1; i < fun->payload.fn.params.count; i--) {
                    gen_push_value_stack(dst_arena, &instructions, fun->payload.fn.params.nodes[i]);
                }

                gen_push_fn_stack(dst_arena, &instructions, callee_to_ptr(ctx, fun));

                instructions = append_nodes(dst_arena, instructions, call_instr(dst_arena, (Call) {
                    .callee = ctx->god_fn,
                    .args = nodes(dst_arena, 0, NULL)
                }));

                new_entry_pt->payload.fn.block = block(dst_arena, (Block) {
                    .instructions = instructions,
                    .terminator = fn_ret(dst_arena, (Return) {
                        .fn = NULL,
                        .values = nodes(dst_arena, 0, NULL)
                    })
                });
            }

            register_processed(&ctx->rewriter, old, fun);
            for (size_t i = 0; i < fun->payload.fn.params.count; i++)
                register_processed(&ctx->rewriter, old->payload.fn.params.nodes[i], fun->payload.fn.params.nodes[i]);
            fun->payload.fn.block = lower_callf_process(ctx, old->payload.fn.block);
            return fun;
        }
        case Block_TAG: {
            // this may miss call instructions...
            Nodes instructions = old->payload.block.instructions;

            const Node* terminator = old->payload.block.terminator;

            switch (terminator->tag) {
                case Return_TAG: {
                    Nodes ret_values = terminator->payload.fn_ret.values;

                    if (ret_values.count > 0) {
                        // Pop the old return address off the stack
                        const Node* ret_tmp_var = gen_pop_fn_stack(dst_arena, &instructions, "return_tmp");
                        // Push the return values as arguments to the return function
                        gen_push_values_stack(dst_arena, &instructions, ret_values);
                        // Push back the return address on the now-top of the stack
                        gen_push_fn_stack(dst_arena, &instructions, ret_tmp_var);
                    }
                    // Kill the function
                    terminator = fn_ret(dst_arena, (Return) {
                        .fn = NULL,
                        .values = nodes(dst_arena, 0, NULL)
                    });
                    break;
                }
                case Callf_TAG: {
                    // put the return address at the bottom
                    gen_push_fn_stack(dst_arena, &instructions, callee_to_ptr(ctx, terminator->payload.callf.ret_fn));
                    // push the arguments to the next call, then the target ptr
                    gen_push_values_stack(dst_arena, &instructions, terminator->payload.callf.args);
                    gen_push_fn_stack(dst_arena, &instructions, callee_to_ptr(ctx, terminator->payload.callf.callee));
                    // Kill the function
                    terminator = fn_ret(dst_arena, (Return) {
                        .fn = NULL,
                        .values = nodes(dst_arena, 0, NULL)
                    });
                    break;
                }
                default: terminator = lower_callf_process(ctx, terminator); break;
            }
            return block(dst_arena, (Block) {
                .instructions = instructions,
                .terminator = terminator
            });
        }
        default: return recreate_node_identity(&ctx->rewriter, old);
    }
}

void generate_top_level_dispatch_fn(Context* ctx, const Node* root, Node* dispatcher_fn) {
    IrArena* dst_arena = ctx->rewriter.dst_arena;

    Nodes loop_body_instructions = nodes(dst_arena, 0, NULL);
    const Node* next_function = gen_pop_fn_stack(dst_arena, &loop_body_instructions, "next_function");

    struct List* literals = new_list(const Node*);
    struct List* cases = new_list(const Node*);

    const Node* zero_lit = int_literal(dst_arena, (IntLiteral) {.value = 0});
    const Node* zero_case = block(dst_arena, (Block) {
        .instructions = nodes(dst_arena, 0, NULL),
        .terminator = merge(dst_arena, (Merge) {
            .args = nodes(dst_arena, 0, NULL),
            .what = Break
        })
    });

    append_list(const Node*, literals, zero_lit);
    append_list(const Node*, cases, zero_case);

    for (size_t i = 0; i < root->payload.root.declarations.count; i++) {
        const Node* decl = root->payload.root.declarations.nodes[i];
        if (decl->tag == Function_TAG) {
            const Node* fn_lit = callee_to_ptr(ctx, find_processed(&ctx->rewriter, decl));

            const FnType* fn_type = &without_qualifier(decl->type)->payload.fn_type;
            Nodes case_instructions = nodes(dst_arena, 0, NULL);
            LARRAY(const Node*, fn_args, fn_type->param_types.count);
            for (size_t j = 0; j < fn_type->param_types.count; j++) {
                fn_args[j] = gen_pop_value_stack(dst_arena, &case_instructions, format_string(dst_arena, "arg_%d", (int) j), without_qualifier(fn_type->param_types.nodes[j]));
            }

            case_instructions = append_nodes(dst_arena, case_instructions, call_instr(dst_arena, (Call) {
                .callee = find_processed(&ctx->rewriter, decl),
                .args = nodes(dst_arena, fn_type->param_types.count, fn_args)
            }));

            const Node* fn_case = block(dst_arena, (Block) {
                .instructions = case_instructions,
                .terminator = merge(dst_arena, (Merge) {
                    .args = nodes(dst_arena, 0, NULL),
                    .what = Continue
                })
            });

            append_list(const Node*, literals, fn_lit);
            append_list(const Node*, cases, fn_case);
        }
    }

    loop_body_instructions = append_nodes(dst_arena, loop_body_instructions, match_instr(dst_arena, (Match) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .inspect = next_function,
        .literals = nodes(dst_arena, entries_count_list(literals), read_list(const Node*, literals)),
        .cases = nodes(dst_arena, entries_count_list(cases), read_list(const Node*, cases)),
        .default_case = block(dst_arena, (Block) {
            .instructions = nodes(dst_arena, 0, NULL),
            .terminator = unreachable(dst_arena)
        })
    }));

    destroy_list(literals);
    destroy_list(cases);

    const Node* loop_body = block(dst_arena, (Block) {
        .instructions = loop_body_instructions,
        .terminator = unreachable(dst_arena)
    });

    Nodes disptcher_body_instructions = nodes(dst_arena, 1, (const Node* []) { loop_instr(dst_arena, (Loop) {
        .yield_types = nodes(dst_arena, 0, NULL),
        .params = nodes(dst_arena, 0, NULL),
        .initial_args = nodes(dst_arena, 0, NULL),
        .body = loop_body
    }) });

    dispatcher_fn->payload.fn.block = block(dst_arena, (Block) {
        .instructions = disptcher_body_instructions,
        .terminator = fn_ret(dst_arena, (Return) {
            .values = nodes(dst_arena, 0, NULL),
            .fn = NULL,
        })
    });
}

const Node* lower_callf(IrArena* src_arena, IrArena* dst_arena, const Node* src_program) {
    struct List* new_decls_list = new_list(const Node*);
    struct Dict* done = new_dict(const Node*, Node*, (HashFn) hash_node, (CmpFn) compare_node);
    struct Dict* ptrs = new_dict(const Node*, FnPtr, (HashFn) hash_node, (CmpFn) compare_node);

    Node* dispatcher_fn = fn(dst_arena, (FnAttributes) {.entry_point_type = NotAnEntryPoint, .is_continuation = false}, "top_dispatcher", nodes(dst_arena, 0, NULL), nodes(dst_arena, 0, NULL));
    append_list(const Node*, new_decls_list, dispatcher_fn);

    Context ctx = {
        .rewriter = {
            .dst_arena = dst_arena,
            .src_arena = src_arena,
            .rewrite_fn = (RewriteFn) lower_callf_process,
            .rewrite_decl_body = NULL,
            .processed = done,
        },
        .assigned_fn_ptrs = ptrs,
        .next_fn_ptr = 1,

        .new_decls = new_decls_list,
        .god_fn = dispatcher_fn,
    };

    const Node* rewritten = recreate_node_identity(&ctx.rewriter, src_program);

    generate_top_level_dispatch_fn(&ctx, src_program, dispatcher_fn);

    Nodes new_decls = rewritten->payload.root.declarations;
    for (size_t i = 0; i < entries_count_list(new_decls_list); i++) {
        new_decls = append_nodes(dst_arena, new_decls, read_list(const Node*, new_decls_list)[i]);
    }
    rewritten = root(dst_arena, (Root) {
        .declarations = new_decls
    });

    destroy_list(new_decls_list);

    destroy_dict(done);
    destroy_dict(ptrs);
    return rewritten;
}
