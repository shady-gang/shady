#ifndef SHADY_META_INTRINSICS_H
#define SHADY_META_INTRINSICS_H

#include "llvm-c/Core.h"
#include "shady_meta.h"

typedef struct ShdIntrinsics_ ShdIntrinsics;

ShdIntrinsics* create_shd_intrinsics(LLVMModuleRef mod);
void destroy_shd_intrinsics(ShdIntrinsics*);

typedef struct {
    shady_meta_instruction meta;
    shady_meta_id defined_id;
    LLVMTypeRef type;
} shady_meta_builtin_type;

typedef struct {
    shady_meta_instruction meta;
    union {
        shady_meta_literal_i32 literal_i32;
        shady_meta_literal_string literal_string;
        shady_meta_builtin_type builtin_type;
        shady_meta_param_ref param_ref;
        shady_meta_ext_op ext_op;
    };
} shady_parsed_meta_instruction;

const shady_parsed_meta_instruction* shd_meta_id_definition(const ShdIntrinsics*, size_t id);
shady_meta_id shd_meta_id_from_name(const ShdIntrinsics*, const char*);

#endif
