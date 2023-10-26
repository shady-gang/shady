#ifndef SHADY_L2S_PRIVATE_H
#define SHADY_L2S_PRIVATE_H

#include "l2s.h"
#include "arena.h"

#include "llvm-c/Core.h"

#include <assert.h>
#include <string.h>

typedef struct {
    LLVMContextRef ctx;
    struct Dict* map;
    struct Dict* annotations;
    struct Dict* scopes;
    Arena* annotations_arena;
    LLVMModuleRef src;
    Module* dst;
} Parser;

#ifndef LLVM_VERSION_MAJOR
#error "Missing LLVM_VERSION_MAJOR"
#else
#define UNTYPED_POINTERS (LLVM_VERSION_MAJOR >= 15)
#endif

typedef struct ParsedAnnotationContents_ {
    const Node* payload;
    struct ParsedAnnotationContents_* next;
} ParsedAnnotation;

ParsedAnnotation* find_annotation(Parser*, const Node*);
ParsedAnnotation* next_annotation(ParsedAnnotation*);
void add_annotation(Parser*, const Node*, ParsedAnnotation);

void process_llvm_annotations(Parser* p, LLVMValueRef global);

const Node* convert_value(Parser* p, LLVMValueRef v);
const Node* convert_function(Parser* p, LLVMValueRef fn);
const Type* convert_type(Parser* p, LLVMTypeRef t);
const Node* convert_metadata(Parser* p, LLVMMetadataRef meta);
const Node* convert_global(Parser* p, LLVMValueRef global);
const Node* convert_function(Parser* p, LLVMValueRef fn);
const Node* convert_basic_block(Parser* p, Node* fn, LLVMBasicBlockRef bb);

typedef struct {
    const Node* terminator;
    const Node* instruction;
    Nodes result_types;
} EmittedInstr;

EmittedInstr convert_instruction(Parser* p, Node* fn_or_bb, BodyBuilder* b, LLVMValueRef instr);

Nodes scope_to_string(Parser* p, LLVMMetadataRef dbgloc);

void postprocess(Parser*, Module* src, Module* dst);

static String is_llvm_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn) || LLVMIsConstant(fn));
    String name = LLVMGetValueName(fn);
    if (memcmp(name, "llvm.", 5) == 0)
        return name;
    return NULL;
}

#endif
