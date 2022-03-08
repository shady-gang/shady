#include "list.h"
#include "dict.h"

#include "../implem.h"
#include "../type.h"

#include "spirv_builder.h"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

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
        case QualType: {
            // SPIR-V does not care about our type qualifiers.
            new = emit_type(emitter, type->payload.qualified.type);
            break;
        }
        default: error("Don't know how to emit type\n")
    }

    insert_dict_and_get_result(struct Type*, SpvId, emitter->type_ids, type, new);
    return new;
}

void emit(const struct Program* program, FILE* output) {
    struct List* words = new_list(uint32_t);

    struct SpvFileBuilder* file_builder = spvb_begin();

    struct SpvEmitter emitter = {
        .file_builder = file_builder,
        .type_ids = new_dict(struct Type*, SpvId, (HashFn) hash_type, (CmpFn) compare_type),
        .node_ids = new_dict(struct Node*, SpvId, (HashFn) hash_node, (CmpFn) compare_node),
    };

    spvb_capability(file_builder, SpvCapabilityShader);
    spvb_capability(file_builder, SpvCapabilityLinkage);

    for (size_t i = 0; i < program->variables.count; i++) {
        const struct Node* variable = program->variables.nodes[i];
        const struct Node* definition = program->definitions.nodes[i];

        enum DivergenceQualifier qual;
        const struct Type* unqualified_type = strip_qualifier(variable->type, &qual);

        switch (unqualified_type->tag) {
            case FnType: {
                assert(qual == Uniform && "top-level functions should never be non-uniform");
                printf("TODO: emit fn\n");
                break;
            } case PtrType: { // this is some global variable
                SpvId id = spvb_global_variable(file_builder, emit_type(&emitter, unqualified_type), emit_addr_space(unqualified_type->payload.ptr.address_space));
                spvb_name(file_builder, id, variable->payload.var.name);
                break;
            } default: { // it must be some global constant
                SHADY_NOT_IMPLEM
            }
        }
    }

    spvb_finish(file_builder, words);
    destroy_dict(emitter.type_ids);
    destroy_dict(emitter.node_ids);

    fwrite(words->alloc, words->elements_count, 4, output);

    destroy_list(words);
}