#include "stdio.h"
#include "ir.h"

void foo() {
    struct IrArena* arena = new_arena();
    destroy_arena(arena);
}

int main() {
    printf("hi\n");
    return 0;
}
