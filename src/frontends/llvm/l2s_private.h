#ifndef SHADY_L2S_PRIVATE_H
#define SHADY_L2S_PRIVATE_H

#include "l2s.h"

#include "llvm-c/Core.h"

#include "assert.h"
#include "string.h"

typedef struct {
    LLVMContextRef ctx;
    struct Dict* map;
    LLVMModuleRef src;
    Module* dst;
    bool untyped_pointers;
} Parser;

typedef struct {
    const Node* terminator;
    const Node* instruction;
    Nodes result_types;
} EmittedInstr;

EmittedInstr emit_instruction(Parser* p, BodyBuilder* b, LLVMValueRef instr);
const Node* convert_value(Parser* p, LLVMValueRef v);
const Node* convert_function(Parser* p, LLVMValueRef fn);
const Type* convert_type(Parser* p, LLVMTypeRef t);
const Node* convert_metadata(Parser* p, LLVMMetadataRef meta);

void postprocess(Module* src, Module* dst);

static String is_llvm_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn) || LLVMIsConstant(fn));
    String name = LLVMGetValueName(fn);
    if (memcmp(name, "llvm.", 5) == 0)
        return name;
    return NULL;
}

#endif
