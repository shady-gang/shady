#ifndef SHADY_EMIT_SPIRV_H
#define SHADY_EMIT_SPIRV_H

#include "shady/ir.h"
#include "emit_spv_builtins.h"

#include "spirv_builder.h"

typedef struct SpvFileBuilder* FileBuilder;
typedef struct SpvFnBuilder* FnBuilder;
typedef struct SpvBasicBlockBuilder* BBBuilder;

typedef struct Emitter_ {
    Module* module;
    IrArena* arena;
    CompilerConfig* configuration;
    FileBuilder file_builder;
    SpvId void_t;
    struct Dict* node_ids;
    struct Dict* bb_builders;
    SpvId emitted_builtins[VulkanBuiltinsCount];
    size_t num_entry_pts;

    struct {
        SpvId debug_printf;
    } non_semantic_imported_instrs;
} Emitter;

typedef struct Phi** Phis;

typedef struct {
    SpvId continue_target, break_target, join_target;
    Phis continue_phis, break_phis, join_phis;
} MergeTargets;

#define emit_decl spv_emit_decl
#define emit_type spv_emit_type
#define emit_value spv_emit_value
#define emit_instruction spv_emit_instruction
#define emit_terminator spv_emit_terminator
#define find_reserved_id spv_find_reserved_id
#define emit_nominal_type_body spv_emit_nominal_type_body

SpvId emit_decl(Emitter*, const Node*);
SpvId emit_type(Emitter*, const Type*);
SpvId emit_value(Emitter*, BBBuilder, const Node*);
SpvId emit_builtin(Emitter*, VulkanBuiltins builtin);
void emit_instruction(Emitter*, FnBuilder, BBBuilder*, MergeTargets*, const Node* instruction, size_t results_count, SpvId results[]);
void emit_terminator(Emitter*, FnBuilder, BBBuilder, MergeTargets, const Node* terminator);

SpvId find_reserved_id(Emitter* emitter, const Node* node);
void register_result(Emitter*, const Node*, SpvId id);

SpvStorageClass emit_addr_space(AddressSpace address_space);
// SPIR-V doesn't have multiple return types, this bridges the gap...
SpvId nodes_to_codom(Emitter* emitter, Nodes return_types);
const Type* normalize_type(Emitter* emitter, const Type* type);
void emit_nominal_type_body(Emitter* emitter, const Type* type, SpvId id);

#endif
