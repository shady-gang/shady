#include <stdint.h>

#define location(i) __attribute__((annotate("shady::location::"#i)))
#define input       __attribute__((address_space(389)))
#define output      __attribute__((address_space(390)))

typedef float vec4     __attribute__((ext_vector_type(4)));

location(0) output vec4 fragColor;

__attribute__((annotate("shady::builtin::FragCoord")))
input vec4 fragCoord;

__attribute__((annotate("shady::entry_point::Fragment")))
void main() {
    int fillx = (((int) fragCoord[0]) / 16) % 2;
    int filly = (((int) fragCoord[1]) / 16) % 2;
    int fill = fillx ^ filly;
    fragColor = (vec4) {fill, fill, 0.0f, 1.0f };
}