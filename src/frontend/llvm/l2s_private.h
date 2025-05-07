#ifndef SHADY_L2S_PRIVATE_H
#define SHADY_L2S_PRIVATE_H

#include "shady/fe/llvm.h"

#include "shady/config.h"

#include "arena.h"
#include "util.h"

#include "llvm-c/Core.h"

#include <assert.h>
#include <string.h>

typedef struct {
    const LLVMFrontendConfig* config;
    LLVMContextRef ctx;
    struct Dict* map;
    struct Dict* annotations;
    Arena* annotations_arena;
    LLVMModuleRef src;
    Module* dst;
} Parser;

typedef struct {
    LLVMBasicBlockRef bb;
    LLVMValueRef instr;
    Node* nbb;
    BodyBuilder* builder;
    bool translated;
} BBParseCtx;

typedef struct {
    Node* fn;
    struct Dict* phis;
    struct Dict* bbs;
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

ParsedAnnotation* l2s_find_annotation(Parser* p, const Node* n);
ParsedAnnotation* next_annotation(ParsedAnnotation*);

void l2s_process_llvm_annotations(Parser* p, LLVMValueRef global);

AddressSpace l2s_convert_llvm_address_space(unsigned);
const Node* l2s_convert_value(Parser* p, LLVMValueRef v);
const Node* l2s_convert_function(Parser* p, LLVMValueRef fn);
const Type* l2s_convert_type(Parser* p, LLVMTypeRef t);
const Node* l2s_convert_metadata(Parser* p, LLVMMetadataRef meta);
const Node* l2s_convert_global(Parser* p, LLVMValueRef global);
const Node* l2s_convert_function(Parser* p, LLVMValueRef fn);
const Node* l2s_convert_basic_block_header(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb);
const Node* l2s_convert_basic_block_body(Parser* p, FnParseCtx* fn_ctx, LLVMBasicBlockRef bb);

void l2s_apply_debug_info(Parser* p, LLVMValueRef, const Node*);

typedef struct {
    struct List* list;
} BBPhis;

typedef struct {
    Node* wrapper;
    Node* src;
    LLVMBasicBlockRef dst;
} JumpTodo;

void convert_jump_finish(Parser* p, FnParseCtx*, JumpTodo todo);
const Node* l2s_convert_instruction(Parser* p, FnParseCtx* fn_ctx, Node* fn_or_bb, BodyBuilder* b, LLVMValueRef instr);

Nodes l2s_scope_to_string(Parser* p, LLVMMetadataRef dbgloc);

void l2s_postprocess(Parser* p, Module* src, Module* dst);

inline static String is_llvm_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn) || LLVMIsConstant(fn));
    String name = LLVMGetValueName(fn);
    if (shd_string_starts_with(name, "llvm."))
        return name;
    return NULL;
}

const Type* l2s_get_param_byval_attr(Parser* p, LLVMValueRef fn, size_t param_index);

inline static String is_shady_intrinsic(LLVMValueRef fn) {
    assert(LLVMIsAFunction(fn) || LLVMIsConstant(fn));
    String name = LLVMGetValueName(fn);
    if (shd_string_starts_with(name, "shady::"))
        return name;
    return NULL;
}

#endif
