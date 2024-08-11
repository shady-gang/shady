#ifndef SHADY_EMIT_SPIRV_H
#define SHADY_EMIT_SPIRV_H

#include "shady/ir.h"
#include "shady/be/spirv.h"

#include "spirv_builder.h"

typedef SpvbFileBuilder* FileBuilder;
typedef SpvbFnBuilder* FnBuilder;
typedef SpvbBasicBlockBuilder* BBBuilder;

typedef struct CFG_ CFG;
typedef struct Scheduler_ Scheduler;

typedef struct Emitter_ {
    Module* module;
    IrArena* arena;
    const CompilerConfig* configuration;
    FileBuilder file_builder;
    SpvId void_t;
    struct Dict* global_node_ids;

    struct Dict* current_fn_node_ids;
    struct Dict* bb_builders;
    CFG* cfg;
    Scheduler* scheduler;

    size_t num_entry_pts;

    struct Dict* extended_instruction_sets;
} Emitter;

typedef SpvbPhi** Phis;

#define emit_decl spv_emit_decl
#define emit_type spv_emit_type

SpvId emit_decl(Emitter*, const Node*);
SpvId emit_type(Emitter*, const Type*);
SpvId spv_emit_value(Emitter*, const Node*);
void spv_emit_terminator(Emitter*, FnBuilder, BBBuilder, const Node* abs, const Node* terminator);

void register_result(Emitter*, bool, const Node*, SpvId id);
SpvId* spv_search_emitted(Emitter* emitter, const Node* node);
SpvId spv_find_reserved_id(Emitter* emitter, const Node* node);

BBBuilder spv_find_basic_block_builder(Emitter* emitter, const Node* bb);

SpvId get_extended_instruction_set(Emitter*, const char*);

SpvStorageClass emit_addr_space(Emitter*, AddressSpace address_space);
// SPIR-V doesn't have multiple return types, this bridges the gap...
SpvId nodes_to_codom(Emitter* emitter, Nodes return_types);
const Type* normalize_type(Emitter* emitter, const Type* type);
void spv_emit_nominal_type_body(Emitter* emitter, const Type* type, SpvId id);

#endif
