#ifndef SHADY_EMIT_SPIRV_H
#define SHADY_EMIT_SPIRV_H

#include "shady/ir.h"
#include "emit_builtins.h"

#include "spirv_builder.h"

typedef struct SpvFileBuilder* FileBuilder;
typedef struct SpvFnBuilder* FnBuilder;
typedef struct SpvBasicBlockBuilder* BBBuilder;

typedef struct Emitter_ {
    IrArena* arena;
    CompilerConfig* configuration;
    FileBuilder file_builder;
    SpvId void_t;
    struct Dict* node_ids;
    SpvId emitted_builtins[VulkanBuiltinsCount];
} Emitter;

SpvStorageClass emit_addr_space(AddressSpace address_space);

SpvId emit_type(Emitter* emitter, const Type* type);
SpvId emit_value(Emitter* emitter, const Node* node, const SpvId* use_id);
SpvId emit_builtin(Emitter* emitter, VulkanBuiltins builtin);

#endif
