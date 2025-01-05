#ifndef SHADY_EMIT_SPIRV_H
#define SHADY_EMIT_SPIRV_H

#include "shady/ir.h"
#include "shady/be/spirv.h"

#include "spirv_builder.h"

typedef struct CFG_ CFG;
typedef struct Scheduler_ Scheduler;

typedef SpvbFileBuilder* FileBuilder;
typedef SpvbBasicBlockBuilder* BBBuilder;

typedef struct {
    SpvbFnBuilder* base;
    CFG* cfg;
    Scheduler* scheduler;
    struct Dict* emitted;
    struct {
        SpvId continue_id;
        BBBuilder continue_builder;
    }* per_bb;
} FnBuilder;

typedef struct Emitter_ {
    Module* module;
    IrArena* arena;
    const CompilerConfig* configuration;
    const SPIRVTargetConfig spirv_tgt;
    FileBuilder file_builder;
    SpvId void_t;
    struct Dict* global_node_ids;

    struct Dict* bb_builders;

    size_t num_entry_pts;
    struct List* interface_vars;

    struct Dict* extended_instruction_sets;
} Emitter;

typedef SpvbPhi** Phis;

SpvId spv_emit_decl(Emitter*, const Node*);
SpvId spv_emit_type(Emitter*, const Type*);
SpvId spv_emit_value(Emitter*, FnBuilder*, const Node*);
SpvId spv_emit_mem(Emitter*, FnBuilder*, const Node*);
void spv_emit_terminator(Emitter*, FnBuilder*, BBBuilder, const Node* abs, const Node* terminator);

void spv_register_emitted(Emitter*, FnBuilder*, const Node*, SpvId id);
SpvId* spv_search_emitted(Emitter* emitter, FnBuilder*, const Node* node);
SpvId spv_find_emitted(Emitter* emitter, FnBuilder*, const Node* node);

BBBuilder spv_find_basic_block_builder(Emitter* emitter, const Node* bb);

SpvId spv_get_extended_instruction_set(Emitter*, const char*);

SpvStorageClass spv_emit_addr_space(Emitter*, AddressSpace address_space);
// SPIR-V doesn't have multiple return types, this bridges the gap...
SpvId spv_types_to_codom(Emitter* emitter, Nodes return_types);
const Type* spv_normalize_type(Emitter* emitter, const Type* type);
void spv_emit_nominal_type_body(Emitter* emitter, const Type* type, SpvId id);
void shd_spv_register_interface(Emitter* emitter, const Node* n, SpvId id);

#endif
