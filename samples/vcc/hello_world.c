#include <shady.h>

void debug_printf(const char*) __asm__("shady::prim_op::debug_printf");

compute_shader local_size(1, 1, 1)
void main() {
    debug_printf("Hello World from Vcc!\n");
}
