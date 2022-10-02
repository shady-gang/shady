#include "emit_c.h"

void emit_pack_code(Emitter* e, Printer* p, const Nodes* src, String dst) {
    for (size_t i = 0; i < src->count; i++) {
        print(p, "\n%s->_%d = %s", dst, emit_value(e, src->nodes[i]), i);
    }
}

void emit_unpack_code(Emitter* e, Printer* p, String src, const Nodes* dst) {
    for (size_t i = 0; i < dst->count; i++) {
        print(p, "\n%s = %s->_%d", emit_value(e, dst->nodes[i]), src, i);
    }
}