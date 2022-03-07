#include "list.h"
#include "dict.h"

#include "../implem.h"

#include "spirv_builder.h"

#include <stdio.h>
#include <stdint.h>

KeyHash hash_type(struct Type**);
bool compare_type(struct Type**, struct Type**);

KeyHash hash_node(struct Node**);
bool compare_node(struct Node**, struct Type**);

struct SpvEmitter {
    struct SpvFileBuilder* file_builder;
    struct Dict* type_ids;
    struct Dict* node_ids;
};

SpvStorageClass emit_addr_space(enum AddressSpace address_space) {
    switch(address_space) {
        case AsGeneric: return SpvStorageClassGeneric;
        case AsPrivate: return SpvStorageClassPrivate;
        case AsShared: return SpvStorageClassCrossWorkgroup;
        case AsGlobal: return SpvStorageClassPhysicalStorageBuffer;
        default: SHADY_UNREACHABLE;
    }
}

SpvId emit_op(struct SpvEmitter* emitter, const struct Node* node);
SpvId emit_type(struct SpvEmitter* emitter, const struct Type* type) {
    SpvId* existing = find_value_dict(struct Type*, SpvId, emitter->type_ids, type);
    if (existing)
        return *existing;
    
    SpvId new;
    switch (type->tag) {
        case Int:
            new = spvb_int_type(emitter->file_builder, 32, true);
            break;
        case PtrType: {
            SpvId pointee = emit_type(emitter, type->payload.ptr.pointed_type);
            SpvStorageClass sc = emit_addr_space(type->payload.ptr.address_space);
            new = spvb_ptr_type(emitter->file_builder, sc, pointee);
            break;
        }
        default:
            exit(667);
    }

    insert_dict_and_get_result(struct Type*, SpvId, emitter->type_ids, type, new);
    return new;
}

void emit(struct Program program, FILE* output) {
    struct List* words = new_list(uint32_t);

    struct SpvFileBuilder* file_builder = spvb_begin();

    struct SpvEmitter emitter = {
        .file_builder = file_builder,
        .type_ids = new_dict(struct Type*, SpvId, (HashFn) hash_type, (CmpFn) compare_type),
        .node_ids = new_dict(struct Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
    };

    spvb_capability(file_builder, SpvCapabilityShader);
    spvb_capability(file_builder, SpvCapabilityLinkage);

    for (size_t i = 0; i < program.declarations_and_definitions.count; i++) {
        const struct Node* node = program.declarations_and_definitions.nodes[i];
        switch (node->tag) {
            case VariableDecl_TAG: {
                SpvId id = spvb_global_variable(file_builder, emit_type(&emitter, node->type), emit_addr_space(node->payload.var_decl.address_space));
                spvb_name(file_builder, id, node->payload.var_decl.variable->payload.var.name);
                break;
            } case Function_TAG:
                break;
            default:
                exit(666);
        }
    }

    spvb_finish(file_builder, words);
    destroy_dict(emitter.type_ids);
    destroy_dict(emitter.node_ids);

    fwrite(words->alloc, words->elements, 4, output);

    destroy_list(words);
}