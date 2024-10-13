#include <shady.h>

void debug_printf(const char*) __asm__("shady::prim_op::debug_printf");
void debug_printfi(const char*, int) __asm__("shady::prim_op::debug_printf::i");

compute_shader local_size(1, 1, 1)
void main() {
    debug_printf("Hello World from Vcc!\n");
    debug_printfi("I can print numbers too: %d!\n", 42);
}
