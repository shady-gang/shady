#include "list.h"
#include "dict.h"

#include "../implem.h"
#include "../type.h"

#include "spirv_builder.h"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

extern const char* node_tags[];

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

struct SpvEmitter {
    IrArena* arena;
    struct SpvFileBuilder* file_builder;
    SpvId void_t;
    struct Dict* node_ids;
};

SpvStorageClass emit_addr_space(AddressSpace address_space) {
    switch(address_space) {
        case AsGeneric: return SpvStorageClassGeneric;
        case AsPrivate: return SpvStorageClassPrivate;
        case AsShared: return SpvStorageClassCrossWorkgroup;
        case AsGlobal: return SpvStorageClassPhysicalStorageBuffer;
        default: SHADY_UNREACHABLE;
    }
}

SpvId emit_type(struct SpvEmitter* emitter, const Type* type);
SpvId emit_value(struct SpvEmitter* emitter, const Node* node, const SpvId* use_id);
SpvId emit_block(struct SpvEmitter* emitter, struct SpvFnBuilder* fnb, Nodes block);

void emit_primop_call(struct SpvEmitter* emitter, struct SpvFnBuilder* fnb, struct SpvBasicBlockBuilder* bbb, const Node* node, SpvId out[]) {
    switch (node->tag) {
        case Call_TAG: SHADY_NOT_IMPLEM;
        case PrimOp_TAG: {
            const Nodes* args = &node->payload.primop.args;
            LARRAY(SpvId, arr, args->count);
            for (size_t i = 0; i < args->count; i++)
                arr[i] = emit_value(emitter, args->nodes[i], NULL);

            SpvId i32_t = emit_type(emitter, int_type(emitter->arena));

            switch (node->payload.primop.op) {
                case add_op: {
                    out[0] = spvb_binop(bbb, SpvOpIAdd, i32_t, arr[0], arr[1]);
                    return;
                }
                default: error("TODO: unhandled op");
            }
        }
        default: error("neither a primop nor a call")
    }
}

void emit_instruction(struct SpvEmitter* emitter, struct SpvFnBuilder* fnb, struct SpvBasicBlockBuilder* bbb, const Node* instruction) {
    switch (instruction->tag) {
        case Let_TAG: {
            const Nodes* variables = &instruction->payload.let.variables;
            LARRAY(SpvId, out, variables->count);
            emit_primop_call(emitter, fnb, bbb, instruction->payload.let.target, out);
            for (size_t i = 0; i < variables->count; i++) {
                spvb_name(emitter->file_builder, out[i], variables->nodes[i]->payload.var.name);
                insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, variables->nodes[i], out[i]);
            }
            break;
        }
        case Return_TAG: {
            const Nodes* ret_values = &instruction->payload.fn_ret.values;
            switch (ret_values->count) {
                case 0: spvb_return_void(bbb); return;
                case 1: spvb_return_value(bbb, emit_value(emitter, ret_values->nodes[0], NULL)); return;
                default: {
                    LARRAY(SpvId, arr, ret_values->count);
                    for (size_t i = 0; i < ret_values->count; i++)
                        arr[i] = emit_value(emitter, ret_values->nodes[i], NULL);
                    SpvId return_that = spvb_composite(bbb, fn_ret_type_id(fnb), ret_values->count, arr);
                    spvb_return_value(bbb, return_that);
                }
            }
            break;
        }
        default: error("TODO: emit instruction");
    }
}

SpvId emit_block(struct SpvEmitter* emitter, struct SpvFnBuilder* fnb, Nodes block) {
    SpvId bbid = spvb_fresh_id(emitter->file_builder);
    struct SpvBasicBlockBuilder* bbb = spvb_begin_bb(fnb, bbid);

    // TODO split the BB if we encounter control flow
    for (size_t i = 0; i < block.count; i++)
        emit_instruction(emitter, fnb, bbb, block.nodes[i]);

    return bbid;
}

SpvId nodes2codom(struct SpvEmitter* emitter, Nodes return_types) {
    switch (return_types.count) {
        case 0: return emitter->void_t;
        case 1: return emit_type(emitter, return_types.nodes[0]);
        default: {
            const Type* codom_ret_type = record_type(emitter->arena, (RecordType) {.members = return_types});
            return emit_type(emitter, codom_ret_type);
        }
    }
}

SpvId emit_value(struct SpvEmitter* emitter, const Node* node, const SpvId* use_id) {
    SpvId* existing = find_value_dict(struct Node*, SpvId, emitter->node_ids, node);
    if (existing)
        return *existing;

    SpvId new = use_id ? *use_id : spvb_fresh_id(emitter->file_builder);
    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, node, new);

    switch (node->tag) {
        case Variable_TAG: error("this variable should have been resolved already");
        case IntLiteral_TAG: {
            SpvId ty = emit_type(emitter, node->yields.nodes[0]);
            uint32_t arr[] = { node->payload.int_literal.value };
            spvb_constant(emitter->file_builder, new, ty, 1, arr);
            break;
        }
        case Function_TAG: {
            const Node* fn_type = derive_fn_type(emitter->arena, &node->payload.fn);

            struct SpvFnBuilder* fn_builder = spvb_begin_fn(emitter->file_builder, new, emit_type(emitter, fn_type), nodes2codom(emitter, node->payload.fn.return_types));
            Nodes params = node->payload.fn.params;
            for (size_t i = 0; i < params.count; i++) {
                SpvId param_id = spvb_parameter(fn_builder, emit_type(emitter, params.nodes[i]->payload.var.type));
                insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, params.nodes[i], param_id);
            }

            emit_block(emitter, fn_builder, node->payload.fn.instructions);
            spvb_define_function(emitter->file_builder, fn_builder);
            break;
        }
        default: error("don't know hot to emit value");
    }
    return new;
}

SpvId emit_type(struct SpvEmitter* emitter, const Type* type) {
    SpvId* existing = find_value_dict(struct Node*, SpvId, emitter->node_ids, type);
    if (existing)
        return *existing;
    
    SpvId new;
    switch (type->tag) {
        case Int_TAG:
            new = spvb_int_type(emitter->file_builder, 32, true);
            break;
        case PtrType_TAG: {
            SpvId pointee = emit_type(emitter, type->payload.ptr_type.pointed_type);
            SpvStorageClass sc = emit_addr_space(type->payload.ptr_type.address_space);
            new = spvb_ptr_type(emitter->file_builder, sc, pointee);
            break;
        }
        case RecordType_TAG: {
            LARRAY(SpvId, members, type->payload.record_type.members.count);
            for (size_t i = 0; i < type->payload.record_type.members.count; i++)
                members[i] = emit_type(emitter, type->payload.record_type.members.nodes[i]);
            new = spvb_struct_type(emitter->file_builder, type->payload.record_type.members.count, members);
            break;
        }
        case FnType_TAG: {
            const FnType * fnt = &type->payload.fn_type;
            LARRAY(SpvId, params, fnt->param_types.count);
            for (size_t i = 0; i < fnt->param_types.count; i++)
                params[i] = emit_type(emitter, fnt->param_types.nodes[i]);

            new = spvb_fn_type(emitter->file_builder, fnt->param_types.count, params, nodes2codom(emitter, fnt->return_types));
            break;
        }
        case QualifiedType_TAG: {
            // SPIR-V does not care about our type qualifiers.
            new = emit_type(emitter, type->payload.qualified_type.type);
            break;
        }
        default: error("Don't know how to emit type\n")
    }

    insert_dict_and_get_result(struct Node*, SpvId, emitter->node_ids, type, new);
    return new;
}

void emit(IrArena* arena, const Node* root_node, FILE* output) {
    const Root* top_level = &root_node->payload.root;
    struct List* words = new_list(uint32_t);

    struct SpvFileBuilder* file_builder = spvb_begin();

    struct SpvEmitter emitter = {
        .arena = arena,
        .file_builder = file_builder,
        .node_ids = new_dict(struct Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
    };

    emitter.void_t = spvb_void_type(emitter.file_builder);

    spvb_capability(file_builder, SpvCapabilityShader);
    spvb_capability(file_builder, SpvCapabilityLinkage);

    LARRAY(SpvId, ids, top_level->variables.count);
    for (size_t i = 0; i < top_level->variables.count; i++) {
        const Node* variable = top_level->variables.nodes[i];
        ids[i] = spvb_fresh_id(file_builder);
        insert_dict_and_get_result(struct Node*, SpvId, emitter.node_ids, variable, ids[i]);
    }

    for (size_t i = 0; i < top_level->variables.count; i++) {
        const Node* definition = top_level->definitions.nodes[i];

        if (definition == NULL) {
            printf("Warning: NULL definition\n");
            continue;
        }

        emit_value(&emitter, definition, &ids[i]);

        /*switch (definition->tag) {
            default: error("we don't know how to emit %s\n", node_tags[definition->tag]);
        }

        assert(variable->payload.var.type);
        DivergenceQualifier qual;
        const Type* unqualified_type = strip_qualifier(variable->payload.var.type, &qual);

        switch (unqualified_type->tag) {
            case FnType_TAG: {
                assert(qual == Uniform && "top-level functions should never be non-uniform");
                printf("TODO: emit fn\n");
                break;
            } case PtrType_TAG: { // this is some global variable
                SpvId id = spvb_global_variable(file_builder, emit_type(&emitter, unqualified_type), emit_addr_space(unqualified_type->payload.ptr_type.address_space));
                spvb_name(file_builder, id, variable->payload.var.name);
                break;
            } default: { // it must be some global constant
                error("idk %s\n", node_tags[unqualified_type->tag]);
            }
        }*/
    }

    spvb_finish(file_builder, words);
    destroy_dict(emitter.node_ids);

    fwrite(words->alloc, words->elements_count, 4, output);

    destroy_list(words);
}