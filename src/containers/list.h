#ifndef SHADY_LIST_H
#define SHADY_LIST_H

#include <stdlib.h>
#include <string.h>

struct List {
    size_t elements;
    size_t space;
    size_t elem_size;
    void* alloc;
};

#define new_list(T) new_list_impl(sizeof(T))

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

#define append_list(T, list, element) append_list_impl(list, (void*) (element), sizeof(T))
void append_list_impl(struct List* list, void* element, size_t element_size) {
    if (list->elements == list->space)
        grow_list(list);
    memcpy(list->alloc + element_size * list->elements, element, element_size);
    list->elements++;
}

#define pop_list(T, list) * ((T*) pop_list_impl(list))
void* pop_list_impl(struct List* list) { \
    void* last = &(list->alloc)[--list->elem_size]; \
    return last; \
}

void add_list(struct List* list, int index);
void remove_list(struct List* list);

#endif
