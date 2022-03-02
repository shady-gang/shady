#include "../implem.h"
#include "../containers/list.h"
#include "spirv_builder.h"

#include <stdio.h>
#include <stdint.h>

void dummy_builder(struct SpvFileBuilder* b) {

}

void emit(struct Program program, FILE* output) {
    struct List* words = new_list(uint32_t);

    spvb_build_file(&dummy_builder, words);

    fwrite(words->alloc, words->elements, 4, output);
    destroy_list(words);
}