#ifndef SHADY_EMIT_C
#define SHADY_EMIT_C

#include "shady/ir.h"
#include "shady/builtins.h"
#include "shady/be/c.h"

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
#define is_term_empty(t) (!t.var && !t.value)

typedef Strings Phis;

typedef struct CompilerConfig_ CompilerConfig;

typedef struct {
    const CompilerConfig* compiler_config;
    CEmitterConfig config;
    IrArena* arena;
    Printer *type_decls, *fn_decls, *fn_defs;
    struct {
        Phis selection, loop_continue, loop_break;
    } phis;

    struct Dict* emitted_terms;
    struct Dict* emitted_types;

    int total_workgroup_size;
    bool use_private_globals;
    Printer* entrypoint_prelude;

    bool need_64b_ext;
} Emitter;

void register_emitted(Emitter*, const Node*, CTerm);
void register_emitted_type(Emitter*, const Type*, String);

CTerm* lookup_existing_term(Emitter* emitter, const Node*);
CType* lookup_existing_type(Emitter* emitter, const Type*);

CValue to_cvalue(Emitter*, CTerm);
CAddr deref_term(Emitter*, CTerm);

void c_emit_decl(Emitter* emitter, const Node* decl);
CType c_emit_type(Emitter* emitter, const Type*, const char* identifier);
String c_emit_fn_head(Emitter* emitter, const Node* fn_type, String center, const Node* fn);
void c_emit_nominal_type_body(Emitter* emitter, String name, const Type* type);

void c_emit_variable_declaration(Emitter* emitter, Printer* block_printer, const Type* t, String variable_name, bool mut, const CTerm* initializer);

CTerm c_emit_value(Emitter* emitter, Printer*, const Node* value);
CTerm c_emit_builtin(Emitter*, Builtin);

String c_legalize_identifier(Emitter*, String);
String c_get_record_field_name(const Type* t, size_t i);
CTerm ispc_varying_ptr_helper(Emitter* emitter, Printer* block_printer, const Type* ptr_type, CTerm term);

CTerm c_bind_intermediary_result(Emitter*, Printer* p, const Type* t, CTerm term);

CTerm c_emit_instruction(Emitter* emitter, Printer* p, const Node* instruction);
void c_emit_terminator(Emitter* emitter, Printer* p, const Node* terminator);
String c_emit_lambda_body(Emitter*, const Node*, const Nodes* nested_basic_blocks);

void c_emit_pack_code(Printer*, Strings, String dst);
void c_emit_unpack_code(Printer*, String src, Strings dst);

#define free_tmp_str(s) free((char*) (s))

inline static bool is_glsl_scalar_type(const Type* t) {
    return t->tag == Bool_TAG || t->tag == Int_TAG || t->tag == Float_TAG;
}

#endif
