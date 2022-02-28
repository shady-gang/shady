#include "list.h"
#include "assert.h"

struct List* new_list_impl(size_t elem_size) {
    struct List* list = (struct List*) malloc(sizeof (struct List));
    *list = (struct List) {
            .elements = 0,
            .space = 8,
            .elem_size = elem_size,
            .alloc = malloc(elem_size * 8)
    };
    return list;
}

void destroy_list(struct List* list) {
    free(list->alloc);
    free(list);
}

void grow_list(struct List* list) {
    list->space *= 2;
    list->alloc = realloc(list->alloc, list->space);
}

void append_list_impl(struct List* list, void* element) {
    if (list->elements == list->space)
        grow_list(list);
    size_t element_size = list->elem_size;
    memcpy((void*) ((size_t) list->alloc + element_size * list->elements), element, element_size);
    list->elements++;
}

void* pop_list_impl(struct List* list) {
    assert(list->elements > 0);
    void* last = (void*) ((size_t)(list->alloc) - list->elem_size);
    return last;
}
