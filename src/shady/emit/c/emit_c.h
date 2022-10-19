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

typedef struct {
    EmitterConfig config;
    IrArena* arena;
    Printer *type_decls, *fn_decls, *fn_defs;
    struct {
        const Nodes *selection, *loop_continue, *loop_break;
    } phis;
    struct Dict* emitted;
} Emitter;

#define emit_type c_emit_type
#define emit_value c_emit_value
#define emit_instruction c_emit_instruction
#define emit_body c_emit_body

String emit_type(Emitter* emitter, const Type* type, const char* identifier);
String emit_value(Emitter* emitter, const Node* value);
void emit_instruction(Emitter* emitter, Printer* p, const Node* instruction, String outputs[]);
String emit_body(Emitter* emitter, const Node* body, const Nodes* bbs);

void emit_pack_code(Emitter*, Printer*, const Nodes* src, String dst);
void emit_unpack_code(Emitter*, Printer*, String src, const Nodes* dst);

#endif
