#ifndef SHADY_FE_LLVM_H
#define SHADY_FE_LLVM_H

#include "shady/ir.h"

#include <stdbool.h>

typedef struct {
    struct {
        bool restructure_with_heuristics;
        bool add_scope_annotations;
        bool has_scope_annotations;
    } input_cf;
} LLVMFrontendConfig;

LLVMFrontendConfig shd_get_default_llvm_frontend_config(void);

void shd_parse_llvm_frontend_args(LLVMFrontendConfig* config, int* pargc, char** argv);

bool shd_parse_llvm(const CompilerConfig*, const LLVMFrontendConfig*, const TargetConfig*, size_t len, const char* data, String name, Module** dst);

int shd_get_linked_major_llvm_version();

#endif
