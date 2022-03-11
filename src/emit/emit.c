#include "list.h"
#include "dict.h"

#include "../implem.h"
#include "../type.h"

#include "spirv_builder.h"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

struct SpvEmitter {
    struct SpvFileBuilder* file_builder;
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

SpvId emit_op(struct SpvEmitter* emitter, const Node* node);
SpvId emit_type(struct SpvEmitter* emitter, const Type* type) {
    SpvId* existing = find_value_dict(struct Type*, SpvId, emitter->node_ids, type);
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
        case QualifiedType_TAG: {
            // SPIR-V does not care about our type qualifiers.
            new = emit_type(emitter, type->payload.qualified_type.type);
            break;
        }
        default: error("Don't know how to emit type\n")
    }

    insert_dict_and_get_result(struct Type*, SpvId, emitter->node_ids, type, new);
    return new;
}

void emit(const Node* root_node, FILE* output) {
    const Root* top_level = &root_node->payload.root;
    struct List* words = new_list(uint32_t);

    struct SpvFileBuilder* file_builder = spvb_begin();

    struct SpvEmitter emitter = {
        .file_builder = file_builder,
        .node_ids = new_dict(struct Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
    };

    spvb_capability(file_builder, SpvCapabilityShader);
    spvb_capability(file_builder, SpvCapabilityLinkage);

    for (size_t i = 0; i < top_level->variables.count; i++) {
        const Node* variable = top_level->variables.nodes[i];
        const Node* definition = top_level->definitions.nodes[i];

        DivergenceQualifier qual;
        const Type* unqualified_type = strip_qualifier(variable->type, &qual);

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
                SHADY_NOT_IMPLEM
            }
        }
    }

    spvb_finish(file_builder, words);
    destroy_dict(emitter.node_ids);

    fwrite(words->alloc, words->elements_count, 4, output);

    destroy_list(words);
}