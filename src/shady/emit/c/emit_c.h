#ifndef SHADY_EMIT_C
#define SHADY_EMIT_C

#include "shady/ir.h"
#include "growy.h"
#include "arena.h"
#include "printer.h"

typedef enum {
    C,
    GLSL
} Dialect;

typedef struct {
    CompilerConfig* config;
    Dialect dialect;
    bool explicitly_sized_types;
} EmitterConfig;

typedef Strings Phis;

typedef struct {
    EmitterConfig config;
    IrArena* arena;
    Printer *type_decls, *fn_decls, *fn_defs;
    struct {
        Phis selection, loop_continue, loop_break;
    } phis;
    struct Dict* emitted;
} Emitter;

#define emit_type c_emit_type
#define emit_value c_emit_value
#define emit_instruction c_emit_instruction
#define emit_lambda_body c_emit_lambda_body

void register_emitted(Emitter*, const Node*, String);
void register_emitted_list(Emitter*, Nodes, Strings);

String emit_type(Emitter* emitter, const Type* type, const char* identifier);
String emit_value(Emitter* emitter, const Node* value);
Strings emit_values(Emitter* emitter, Nodes);
Strings emit_variable_declarations(Emitter* emitter, Printer* p, Nodes vars);
void emit_instruction(Emitter* emitter, Printer* p, const Node* instruction, Strings outputs);
String emit_lambda_body   (Emitter*,           const Node*, const Nodes* nested_basic_blocks);
void   emit_lambda_body_at(Emitter*, Printer*, const Node*, const Nodes* nested_basic_blocks);

void emit_pack_code(Emitter*, Printer*, const Nodes* src, String dst);
void emit_unpack_code(Emitter*, Printer*, String src, Strings dst);

#endif
