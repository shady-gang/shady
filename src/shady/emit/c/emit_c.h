#ifndef SHADY_EMIT_C
#define SHADY_EMIT_C

#include "shady/ir.h"
#include "shady/builtins.h"
#include "growy.h"
#include "arena.h"
#include "printer.h"

/// SSA-like things, you can read them
typedef String CValue;
/// non-SSA like things, they represent addresses
typedef String CAddr;

typedef String CType;

typedef struct {
    CValue value;
    CAddr var;
} CTerm;

#define term_from_cvalue(t) (CTerm) { .value = t }
#define term_from_cvar(t) (CTerm) { .var = t }
#define empty_term() (CTerm) { 0 }

typedef Strings Phis;

typedef struct {
    CEmitterConfig config;
    IrArena* arena;
    Printer *type_decls, *fn_decls, *fn_defs;
    struct {
        Phis selection, loop_continue, loop_break;
    } phis;

    struct Dict* emitted_terms;
    struct Dict* emitted_types;
} Emitter;

CTerm register_emitted_cterm(Emitter* emitter, const Node* node, CTerm as);
CType register_emitted_ctype(Emitter* emitter, const Node* node, CType as);

CTerm* lookup_existing_cterm(Emitter* emitter, const Node* node);
CType* lookup_existing_ctype(Emitter* emitter, const Type* node);

CValue to_cvalue(Emitter*, CTerm);
CAddr deref_cterm(Emitter*, CTerm);

void c_emit_decl(Emitter* emitter, const Node* decl);
CType c_emit_type(Emitter* emitter, const Type*, const char* identifier);
String c_emit_fn_head(Emitter* emitter, const Node* fn_type, String center, const Node* fn);
void c_emit_nominal_type_body(Emitter* emitter, String name, const Type* type);

CTerm c_emit_value(Emitter* emitter, Printer*, const Node* value);
CTerm c_emit_builtin(Emitter*, Builtin);

String legalize_c_identifier(Emitter*, String);
String c_get_record_field_name(const Type* t, size_t i);
CTerm ispc_varying_ptr_helper(Emitter* emitter, Printer* block_printer, const Type* ptr_type, CTerm term);

void c_emit_terminator(Emitter*, Printer* block_printer, const Node* terminator);
CTerm c_emit_instruction(Emitter*, Printer* p, const Node* instruction);
void c_emit_lambda_body_at(Emitter*, Printer*, const Node*, const Nodes* nested_basic_blocks);
String c_emit_lambda_body(Emitter*, const Node*, const Nodes* nested_basic_blocks);

// void emit_pack_code(Printer*, Strings, String dst);
// void emit_unpack_code(Printer*, String src, Strings dst);

#define free_tmp_str(s) free((char*) (s))

inline static bool is_glsl_scalar_type(const Type* t) {
    return t->tag == Bool_TAG || t->tag == Int_TAG || t->tag == Float_TAG;
}

#endif
