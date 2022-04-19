#include "type.h"

#include "log.h"
#include "local_array.h"

#include "dict.h"

#include <string.h>
#include <assert.h>

bool is_type(const Node* node) {
    switch (node->tag) {
#define NODEDEF(_, _2, _3, name, _4) case name##_TAG:
TYPE_NODES()
#undef NODEDEF
                 return true;
        default: return false;
    }
}

bool is_subtype(const Type* supertype, const Type* type) {
    if (supertype->tag != type->tag)
        return false;
    switch (supertype->tag) {
        case QualifiedType_TAG: {
            // uniform T <: varying T
            if (supertype->payload.qualified_type.is_uniform && !type->payload.qualified_type.is_uniform)
                return false;
            return is_subtype(supertype->payload.qualified_type.type, type->payload.qualified_type.type);
        }
        case RecordType_TAG: {
            const Nodes* supermembers = &supertype->payload.record_type.members;
            const Nodes* members = &type->payload.record_type.members;
            for (size_t i = 0; i < members->count; i++) {
                if (!is_subtype(supermembers->nodes[i], members->nodes[i]))
                    return false;
            }
            return true;
        }
        case FnType_TAG:
            if (supertype->payload.fn.is_continuation != type->payload.fn.is_continuation)       return false;
            // check returns
            if (supertype->payload.fn.return_types.count != type->payload.fn.return_types.count) return false;
            for (size_t i = 0; i < type->payload.fn.return_types.count; i++)
                if (!is_subtype(supertype->payload.fn.return_types.nodes[i], type->payload.fn.return_types.nodes[i]))
                    return false;
            // check params
            const Nodes* superparams = &supertype->payload.fn_type.param_types;
            const Nodes* params = &type->payload.fn_type.param_types;
            if (params->count != superparams->count) return false;
            for (size_t i = 0; i < params->count; i++) {
                if (!is_subtype(params->nodes[i], superparams->nodes[i]))
                    return false;
            }
            return true;
        case PtrType_TAG: SHADY_NOT_IMPLEM;
        // simple types without a payload
        default: return true;
    }
    SHADY_UNREACHABLE;
}

void check_subtype(const Type* supertype, const Type* type) {
    if (!is_subtype(supertype, type)) {
        print_node(type);
        printf(" isn't a subtype of ");
        print_node(supertype);
        printf("\n");
        error("failed check_subtype")
    }
}

const Type* strip_qualifier(const Type* type, DivergenceQualifier* qual_out) {
    if (type->tag == QualifiedType_TAG) {
        *qual_out = type->payload.qualified_type.is_uniform ? Uniform : Varying;
        return type->payload.qualified_type.type;
    } else {
        *qual_out = Unknown;
        return type;
    }
}

DivergenceQualifier get_qualifier(const Type* type) {
    DivergenceQualifier result;
    strip_qualifier(type, &result);
    return result;
}

const Type* without_qualifier(const Type* type) {
    DivergenceQualifier dontcare;
    return strip_qualifier(type, &dontcare);
}

const Type* check_qualifier_uniform(const Type* type) {
    DivergenceQualifier docare;
    const Type* stripped = strip_qualifier(type, &docare);
    if (docare != Uniform) {
        error_node(type);
        error("this type should be qualified with uniform");
    }
    return stripped;
}

Nodes extract_variable_types(IrArena* arena, const Nodes* variables) {
    LARRAY(const Type*, arr, variables->count);
    for (size_t i = 0; i < variables->count; i++)
        arr[i] = variables->nodes[i]->payload.var.type;
    return nodes(arena, variables->count, arr);
}

const Type* derive_fn_type(IrArena* arena, const Function* fn) {
    return fn_type(arena, (FnType) { .is_continuation = fn->is_continuation, .param_types = extract_variable_types(arena, &fn->params), .return_types = fn->return_types });
}

Nodes check_call(IrArena* arena, const Node* callee, size_t argsc, const Node* args[]) {
    const Type* callee_type = without_qualifier(callee->type);
    if (callee_type->tag != FnType_TAG)
        error("Callees must have a function type");
    if (callee_type->payload.fn_type.param_types.count != argsc)
        error("Mismatched argument counts");
    for (size_t i = 0; i < argsc; i++) {
        const Node* arg = args[i];
        assert(arg && arg->type);
        if (!is_subtype(callee_type->payload.fn_type.param_types.nodes[i], arg->type)) {
            error_print("calle is :");
            print_node(callee);
            error_print("arg #%zu is: \n", i);
            print_node(arg);
            error_print("\n");
            error("Incorrect argument type for argument %zu", i);
        }
    }
    return callee_type->payload.fn_type.return_types;
}

const Type* check_type_fn(IrArena* arena, Function fn) {
    assert(!fn.is_continuation || fn.return_types.count == 0);
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = derive_fn_type(arena, &fn)
    });
}

const Type* check_type_var_decl(IrArena* arena, VariableDecl decl) {
    SHADY_NOT_IMPLEM
    //return ptr_type(arena, (PtrType) { .address_space = decl.address_space, .pointed_type = decl.variable->type });
}

const Type* check_type_var(IrArena* arena, Variable variable) {
    assert(variable.type);
    assert(get_qualifier(variable.type) != Unknown);
    return variable.type;
}

const Type* check_type_untyped_number(IrArena* arena, UntypedNumber untyped) {
    error("should never happen");
}

const Type* check_type_int_literal(IrArena* arena, IntLiteral lit) {
    return qualified_type(arena, (QualifiedType) {
        .is_uniform = true,
        .type = int_type(arena)
    });
}

const Type* check_type_true_lit(IrArena* arena) { return bool_type(arena); }
const Type* check_type_false_lit(IrArena* arena) { return bool_type(arena); }

const Type* check_type_let(IrArena* arena, Let let) {
    Nodes var_tys = extract_variable_types(arena, &let.variables);

    // todo check inputs

    Nodes output_types;
    if (let.op == call_op) {
        const Node* callee = let.args.nodes[0];
        assert(get_qualifier(callee->type) == Uniform);
        output_types = without_qualifier(callee->type)->payload.fn_type.return_types;
    } else
        output_types = op_yields(arena, let.op, let.args);

    // check outputs
    if (output_types.count != var_tys.count)
        error("let variables count != yield count from operation")
    for (size_t i = 0; i < var_tys.count; i++)
        check_subtype(var_tys.nodes[i], output_types.nodes[i]);
    return NULL;
}

const Type* check_type_jump(IrArena* arena, Jump jump) {
    assert(get_qualifier(jump.target->type) == Uniform);
    assert(without_qualifier(jump.target->type)->tag == FnType_TAG);
    const FnType* tgt_type = &without_qualifier(jump.target->type)->payload.fn_type;
    assert(tgt_type->is_continuation);
    for (size_t i = 0; i < tgt_type->param_types.count; i++)
        check_subtype(tgt_type->param_types.nodes[i], jump.args.nodes[i]->type);
    return NULL;
}

const Type* check_type_branch(IrArena* arena, Branch branch) {
    // TODO uniform jump or structured jump
    // assert(get_qualifier(branch.condition->type) == Uniform);

    // branches should be agreed upon
    assert(get_qualifier(branch.true_target->type) == Uniform);
    assert(without_qualifier(branch.true_target->type)->tag == FnType_TAG);
    assert(get_qualifier(branch.false_target->type) == Uniform);
    assert(without_qualifier(branch.false_target->type)->tag == FnType_TAG);

    // TODO check arguments and that both branches match
    return NULL;
}

const Type* check_type_callf(IrArena* arena, Callf callf) {
    // todo some rules on uniformity might be in order here
    const Type* unqual_ret_cont_type = check_qualifier_uniform(callf.ret_cont->type);

    if (unqual_ret_cont_type->tag != FnType_TAG)
        error("callf's first argument must be a function type");
    const FnType* join_cont_type = &unqual_ret_cont_type->payload.fn_type;

    // actually let's not check that because at some point we'll turn those conts into first-class functions
    // as part of the strategy for implementing calls
    // if (join_cont_type->is_continuation)

    const Nodes returned_types = check_call(arena, callf.target, callf.args.count, callf.args.nodes);
    if (join_cont_type->param_types.count != returned_types.count)
        error("the callf return domain does not match the callee codomain");
    for (size_t i = 0; i < join_cont_type->param_types.count; i++) {
        if (!is_subtype(join_cont_type->param_types.nodes[i], returned_types.nodes[i]))
            error("the callf return domain does not match the callee codomain");
    }

    return NULL;
}

const Type* check_type_fn_ret(IrArena* arena, Return ret) {
    // TODO check it then !
    return NULL;
}

const Type* check_type_if_instr(IrArena* arena, IfInstr sel) {
    if (without_qualifier(sel.condition->type) != bool_type(arena))
        error("condition of a selection should be bool");
    return NULL;
}

const Type* check_type_root(IrArena* arena, Root program) {
    return NULL;
}

// TODO handle parameters
Nodes op_params(IrArena* arena, Op op, Nodes args) {
    switch (op) {
        case add_op:
        case sub_op: return nodes(arena, 2, (const Type*[]){ int_type(arena), int_type(arena) });
        case call_op: {
            assert(args.count >= 1);
            return check_call(arena, args.nodes[0], args.count - 1, &args.nodes[1]);
        }
        default: error("unhandled op params");
    }
}

#define empty() nodes(arena, 0, NULL)
#define singleton(t) singleton_impl(arena, t)
Nodes singleton_impl(IrArena* arena, const Type* type) {
    const Type* arr[] = { type };
    return nodes(arena, 1, arr);
}

Nodes op_yields(IrArena* arena, Op op, Nodes args) {
    switch (op) {
        case add_op:
        case sub_op: {
             bool is_result_uniform = true;
             for (size_t i = 0; i < args.count; i++) {
                 const Node* arg = args.nodes[i];
                 DivergenceQualifier op_div = get_qualifier(arg->type);
                 assert(op_div != Unknown); // we expect all operands to be clearly known !
                 is_result_uniform ^= op_div == Uniform;
             }

            return singleton(qualified_type(arena, (QualifiedType) { .is_uniform = is_result_uniform, .type = int_type(arena) }));
        }
        default: error("unhandled op yield");
    }
}
