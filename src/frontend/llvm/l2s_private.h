#ifndef SHADY_L2S_PRIVATE_H
#define SHADY_L2S_PRIVATE_H

#include "l2s.h"

#include "shady/config.h"

#include "arena.h"
#include "util.h"

#include "llvm-c/Core.h"

#include <assert.h>
#include <string.h>

typedef struct {
    const CompilerConfig* config;
    LLVMContextRef ctx;
    struct Dict* map;
    struct Dict* annotations;
    Arena* annotations_arena;
    LLVMModuleRef src;
    Module* dst;
} Parser;

typedef struct {
    Node* fn;
    struct Dict* phis;
    struct List* jumps_todo;
} FnParseCtx;

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

AddressSpace convert_llvm_address_space(unsigned);
const Node* convert_value(Parser* p, LLVMValueRef v);
const Node* convert_function(Parser* p, LLVMValueRef fn);
const Type* convert_type(Parser* p, LLVMTypeRef t);
const Node* convert_metadata(Parser* p, LLVMMetadataRef meta);
const Node* convert_global(Parser* p, LLVMValueRef global);
const Node* convert_function(Parser* p, LLVMValueRef fn);
const Node* convert_basic_block(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb);

typedef struct {
    struct List* list;
} BBPhis;

typedef struct {
    Node* wrapper;
    Node* src;
    LLVMBasicBlockRef dst;
} JumpTodo;

void convert_jump_finish(Parser* p, FnParseCtx*, JumpTodo todo);
const Node* convert_instruction(Parser* p, FnParseCtx*, Node* fn_or_bb, BodyBuilder* b, LLVMValueRef instr);

Nodes scope_to_string(Parser* p, LLVMMetadataRef dbgloc);

void postprocess(Parser*, Module* src, Module* dst);

inline static String is_llvm_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn) || LLVMIsConstant(fn));
    String name = LLVMGetValueName(fn);
    if (string_starts_with(name, "llvm."))
        return name;
    return NULL;
}

inline static String is_shady_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn) || LLVMIsConstant(fn));
    String name = LLVMGetValueName(fn);
    if (string_starts_with(name, "shady::"))
        return name;
    return NULL;
}

#endif
