#include "ir_private.h"
#include "type.h"
#include "log.h"
#include "fold.h"
#include "portability.h"

#include "dict.h"
#include "visit.h"

#include <string.h>
#include <assert.h>

Strings import_strings(IrArena*, Strings);
bool compare_nodes(Nodes* a, Nodes* b);

static void pre_construction_validation(IrArena* arena, Node* node);

const Node* fold_node_operand(NodeTag tag, NodeClass, String, const Node* op);

const Type* check_type_generated(IrArena* a, const Node* node);

static Node* create_node_helper(IrArena* arena, Node node, bool* pfresh) {
    pre_construction_validation(arena, &node);
    if (arena->config.check_types)
        node.type = check_type_generated(arena, &node);

    if (pfresh)
        *pfresh = false;

    Node* ptr = &node;
    Node** found = find_key_dict(Node*, arena->node_set, ptr);
    // sanity check nominal nodes to be unique, check for duplicates in structural nodes
    if (is_nominal(&node))
        assert(!found);
    else if (found)
        return *found;

    if (pfresh)
        *pfresh = true;

    if (arena->config.allow_fold) {
        Node* folded = (Node*) fold_node(arena, ptr);
        if (folded != ptr) {
            // The folding process simplified the node, we store a mapping to that simplified node and bail out !
            insert_set_get_result(Node*, arena->node_set, folded);
            return folded;
        }
    }

    if (arena->config.check_types && node.type)
        assert(is_type(node.type));

    // place the node in the arena and return it
    Node* alloc = (Node*) arena_alloc(arena->arena, sizeof(Node));
    *alloc = node;
    alloc->id = allocate_node_id(arena, alloc);
    insert_set_get_result(const Node*, arena->node_set, alloc);

    return alloc;
}

#include "constructors_generated.c"

Node* param(IrArena* arena, const Type* type, const char* name) {
    Param param = {
        .type = type,
        .name = string(arena, name),
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Param_TAG,
        .payload.param = param
    };
    return create_node_helper(arena, node, NULL);
}

const Node* bind_identifiers(IrArena* arena, const Node* value, const Node* mem, bool mut, Strings names, Nodes types) {
    BindIdentifiers payload = {
        .value = value,
        .mutable = mut,
        .names = names,
        .types = types,
        .mem = mem,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = NULL,
        .tag = BindIdentifiers_TAG,
        .payload.bind_identifiers = payload
    };
    return create_node_helper(arena, node, NULL);
}

const Node* composite_helper(IrArena* a, const Type* t, Nodes contents) {
    return composite(a, (Composite) { .type = t, .contents = contents });
}

const Node* tuple_helper(IrArena* a, Nodes contents) {
    const Type* t = NULL;
    if (a->config.check_types) {
        // infer the type of the tuple
        Nodes member_types = get_values_types(a, contents);
        t = record_type(a, (RecordType) {.members = strip_qualifiers(a, member_types)});
    }

    return composite_helper(a, t, contents);
}

const Node* fn_addr_helper(IrArena* a, const Node* fn) {
    return fn_addr(a, (FnAddr) { .fn = fn });
}

const Node* ref_decl_helper(IrArena* a, const Node* decl) {
    return ref_decl(a, (RefDecl) { .decl = decl });
}

const Node* type_decl_ref_helper(IrArena* a, const Node* decl) {
    return type_decl_ref(a, (TypeDeclRef) { .decl = decl });
}

Node* function(Module* mod, Nodes params, const char* name, Nodes annotations, Nodes return_types) {
    assert(!mod->sealed);
    IrArena* arena = mod->arena;
    Function payload = {
        .module = mod,
        .params = params,
        .body = NULL,
        .name = name,
        .annotations = annotations,
        .return_types = return_types,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Function_TAG,
        .payload.fun = payload
    };
    Node* fn = create_node_helper(arena, node, NULL);
    register_decl_module(mod, fn);

    for (size_t i = 0; i < params.count; i++) {
        Node* param = (Node*) params.nodes[i];
        assert(param->tag == Param_TAG);
        assert(!param->payload.param.abs);
        param->payload.param.abs = fn;
        param->payload.param.pindex = i;
    }

    return fn;
}

Node* basic_block(IrArena* arena, Nodes params, const char* name) {
    BasicBlock payload = {
        .params = params,
        .body = NULL,
        .name = name,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = BasicBlock_TAG,
        .payload.basic_block = payload
    };

    Node* bb = create_node_helper(arena, node, NULL);

    for (size_t i = 0; i < params.count; i++) {
        Node* param = (Node*) params.nodes[i];
        assert(param->tag == Param_TAG);
        assert(!param->payload.param.abs);
        param->payload.param.abs = bb;
        param->payload.param.pindex = i;
    }

    return bb;
}

Node* constant(Module* mod, Nodes annotations, const Type* hint, String name) {
    IrArena* arena = mod->arena;
    Constant cnst = {
        .annotations = annotations,
        .name = string(arena, name),
        .type_hint = hint,
        .value = NULL,
    };
    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = Constant_TAG,
        .payload.constant = cnst
    };
    Node* decl = create_node_helper(arena, node, NULL);
    register_decl_module(mod, decl);
    return decl;
}

Node* global_var(Module* mod, Nodes annotations, const Type* type, const char* name, AddressSpace as) {
    const Node* existing = get_declaration(mod, name);
    if (existing) {
        assert(existing->tag == GlobalVariable_TAG);
        assert(existing->payload.global_variable.type == type);
        assert(existing->payload.global_variable.address_space == as);
        assert(!mod->arena->config.check_types || compare_nodes((Nodes*) &existing->payload.global_variable.annotations, &annotations));
        return (Node*) existing;
    }

    IrArena* arena = mod->arena;
    GlobalVariable gvar = {
        .annotations = annotations,
        .name = string(arena, name),
        .type = type,
        .address_space = as,
        .init = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .tag = GlobalVariable_TAG,
        .payload.global_variable = gvar
    };
    Node* decl = create_node_helper(arena, node, NULL);
    register_decl_module(mod, decl);
    return decl;
}

Type* nominal_type(Module* mod, Nodes annotations, String name) {
    IrArena* arena = mod->arena;
    NominalType payload = {
        .name = string(arena, name),
        .module = mod,
        .annotations = annotations,
        .body = NULL,
    };

    Node node;
    memset((void*) &node, 0, sizeof(Node));
    node = (Node) {
        .arena = arena,
        .type = NULL,
        .tag = NominalType_TAG,
        .payload.nom_type = payload
    };
    Node* decl = create_node_helper(arena, node, NULL);
    register_decl_module(mod, decl);
    return decl;
}

const Node* prim_op_helper(IrArena* a, Op op, Nodes types, Nodes operands) {
    return prim_op(a, (PrimOp) {
        .op = op,
        .type_arguments = types,
        .operands = operands
    });
}

const Node* jump_helper(IrArena* a, const Node* dst, Nodes args, const Node* mem) {
    return jump(a, (Jump) {
        .target = dst,
        .args = args,
        .mem = mem,
    });
}

const Node* unit_type(IrArena* arena) {
     return record_type(arena, (RecordType) {
         .members = empty(arena),
     });
}

const Node* empty_multiple_return_type(IrArena* arena) {
    return qualified_type_helper(unit_type(arena), true);
}

const Node* annotation_value_helper(IrArena* a, String n, const Node* v) {
    return annotation_value(a, (AnnotationValue) { .name = n, .value = v});
}

const Node* string_lit_helper(IrArena* a, String s) {
    return string_lit(a, (StringLiteral) { .string = s });
}

const Type* int_type_helper(IrArena* a, bool s, IntSizes w) { return int_type(a, (Int) { .width = w, .is_signed = s }); }

const Type* int8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = true }); }
const Type* int16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = true }); }
const Type* int32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = true }); }
const Type* int64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = true }); }

const Type* uint8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = false }); }
const Type* uint16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = false }); }
const Type* uint32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = false }); }
const Type* uint64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = false }); }

const Type* int8_literal (IrArena* arena, int8_t  i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,  .value = (uint64_t)  (uint8_t) i, .is_signed = true }); }
const Type* int16_literal(IrArena* arena, int16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value = (uint64_t) (uint16_t) i, .is_signed = true }); }
const Type* int32_literal(IrArena* arena, int32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value = (uint64_t) (uint32_t) i, .is_signed = true }); }
const Type* int64_literal(IrArena* arena, int64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value = (uint64_t) i, .is_signed = true }); }

const Type* uint8_literal (IrArena* arena, uint8_t  i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,  .value = (int64_t) i }); }
const Type* uint16_literal(IrArena* arena, uint16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value = (int64_t) i }); }
const Type* uint32_literal(IrArena* arena, uint32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value = (int64_t) i }); }
const Type* uint64_literal(IrArena* arena, uint64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value = i }); }

const Type* fp16_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy16 }); }
const Type* fp32_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy32 }); }
const Type* fp64_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy64 }); }

const Node* fp_literal_helper(IrArena* a, FloatSizes size, double value) {
    switch (size) {
        case FloatTy16: assert(false); break;
        case FloatTy32: {
            float f = value;
            uint64_t bits = 0;
            memcpy(&bits, &f, sizeof(f));
            return float_literal(a, (FloatLiteral) { .width = size, .value = bits });
        }
        case FloatTy64: {
            uint64_t bits = 0;
            memcpy(&bits, &value, sizeof(value));
            return float_literal(a, (FloatLiteral) { .width = size, .value = bits });
        }
    }
}

const Node* extract_helper(const Node* composite, const Node* index) {
    IrArena* a = composite->arena;
    return prim_op_helper(a, extract_op, empty(a), mk_nodes(a, composite, index));
}

const Node* maybe_tuple_helper(IrArena* a, Nodes values) {
    if (values.count == 1)
        return first(values);
    return tuple_helper(a, values);
}
