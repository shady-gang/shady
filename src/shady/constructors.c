#include "ir_private.h"
#include "type.h"
#include "log.h"
#include "fold.h"
#include "portability.h"

#include "dict.h"
#include "shady/visit.h"

#include <string.h>
#include <assert.h>

Strings _shd_import_strings(IrArena* dst_arena, Strings old_strings);
bool shd_compare_nodes(Nodes* a, Nodes* b);

static void pre_construction_validation(IrArena* arena, Node* node);

const Node* _shd_fold_node_operand(NodeTag tag, NodeClass nc, String opname, const Node* op);

const Type* _shd_check_type_generated(IrArena* a, const Node* node);

Node* _shd_create_node_helper(IrArena* arena, Node node, bool* pfresh) {
    pre_construction_validation(arena, &node);
    if (arena->config.check_types)
        node.type = _shd_check_type_generated(arena, &node);

    if (pfresh)
        *pfresh = false;

    Node* ptr = &node;
    Node** found = shd_dict_find_key(Node*, arena->node_set, ptr);
    // sanity check nominal nodes to be unique, check for duplicates in structural nodes
    if (is_nominal(&node))
        assert(!found);
    else if (found)
        return *found;

    if (pfresh)
        *pfresh = true;

    if (arena->config.allow_fold) {
        Node* folded = (Node*) _shd_fold_node(arena, ptr);
        if (folded != ptr) {
            // The folding process simplified the node, we store a mapping to that simplified node and bail out !
            shd_set_insert_get_result(Node*, arena->node_set, folded);
            return folded;
        }
    }

    if (arena->config.check_types && node.type)
        assert(is_type(node.type));

    // place the node in the arena and return it
    Node* alloc = (Node*) shd_arena_alloc(arena->arena, sizeof(Node));
    *alloc = node;
    alloc->id = _shd_allocate_node_id(arena, alloc);
    shd_set_insert_get_result(const Node*, arena->node_set, alloc);

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
    return _shd_create_node_helper(arena, node, NULL);
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
    Node* fn = _shd_create_node_helper(arena, node, NULL);
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

    Node* bb = _shd_create_node_helper(arena, node, NULL);

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
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    register_decl_module(mod, decl);
    return decl;
}

Node* global_var(Module* mod, Nodes annotations, const Type* type, const char* name, AddressSpace as) {
    const Node* existing = get_declaration(mod, name);
    if (existing) {
        assert(existing->tag == GlobalVariable_TAG);
        assert(existing->payload.global_variable.type == type);
        assert(existing->payload.global_variable.address_space == as);
        assert(!mod->arena->config.check_types || shd_compare_nodes((Nodes*) &existing->payload.global_variable.annotations, &annotations));
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
    Node* decl = _shd_create_node_helper(arena, node, NULL);
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
    Node* decl = _shd_create_node_helper(arena, node, NULL);
    register_decl_module(mod, decl);
    return decl;
}

const Node* lea_helper(IrArena* a, const Node* ptr, const Node* offset, Nodes indices) {
    const Node* lea = ptr_array_element_offset(a, (PtrArrayElementOffset) {
        .ptr = ptr,
        .offset = offset,
    });
    for (size_t i = 0; i < indices.count; i++) {
        lea = ptr_composite_element(a, (PtrCompositeElement) {
            .ptr = lea,
            .index = indices.nodes[i],
        });
    }
    return lea;
}

const Node* unit_type(IrArena* arena) {
     return record_type(arena, (RecordType) {
         .members = shd_empty(arena),
     });
}

const Node* empty_multiple_return_type(IrArena* arena) {
    return shd_as_qualified_type(unit_type(arena), true);
}

const Type* shd_int_type_helper(IrArena* a, bool s, IntSizes w) { return int_type(a, (Int) { .width = w, .is_signed = s }); }

const Type* shd_int8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = true }); }
const Type* shd_int16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = true }); }
const Type* shd_int32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = true }); }
const Type* shd_int64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = true }); }

const Type* shd_uint8_type(IrArena* arena) {  return int_type(arena, (Int) { .width = IntTy8 , .is_signed = false }); }
const Type* shd_uint16_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy16, .is_signed = false }); }
const Type* shd_uint32_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy32, .is_signed = false }); }
const Type* shd_uint64_type(IrArena* arena) { return int_type(arena, (Int) { .width = IntTy64, .is_signed = false }); }

const Node* shd_int8_literal (IrArena* arena, int8_t  i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,  .value = (uint64_t)  (uint8_t) i, .is_signed = true }); }
const Node* shd_int16_literal(IrArena* arena, int16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value = (uint64_t) (uint16_t) i, .is_signed = true }); }
const Node* shd_int32_literal(IrArena* arena, int32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value = (uint64_t) (uint32_t) i, .is_signed = true }); }
const Node* shd_int64_literal(IrArena* arena, int64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value = (uint64_t) i, .is_signed = true }); }

const Node* shd_uint8_literal (IrArena* arena, uint8_t  i) { return int_literal(arena, (IntLiteral) { .width = IntTy8,  .value = (int64_t) i }); }
const Node* shd_uint16_literal(IrArena* arena, uint16_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy16, .value = (int64_t) i }); }
const Node* shd_uint32_literal(IrArena* arena, uint32_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy32, .value = (int64_t) i }); }
const Node* shd_uint64_literal(IrArena* arena, uint64_t i) { return int_literal(arena, (IntLiteral) { .width = IntTy64, .value = i }); }

const Type* shd_fp16_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy16 }); }
const Type* shd_fp32_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy32 }); }
const Type* shd_fp64_type(IrArena* arena) { return float_type(arena, (Float) { .width = FloatTy64 }); }

const Node* shd_fp_literal_helper(IrArena* a, FloatSizes size, double value) {
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
    return prim_op_helper(a, extract_op, shd_empty(a), mk_nodes(a, composite, index));
}

const Node* maybe_tuple_helper(IrArena* a, Nodes values) {
    if (values.count == 1)
        return shd_first(values);
    return tuple_helper(a, values);
}
