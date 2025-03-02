#include <stdint.h>
#include <shady.h>

typedef struct {
    int arr[8];
} S;

S f(S s) {
    S t = s;
    for (int i = 0; i < 8; i++)
        t.arr[i] = s.arr[7 - i];
    return t;
}

compute_shader local_size(32, 1, 1) void main(S s) {
    f(s);
}