#include "list.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

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

size_t entries_count_list(struct List* list) {
    return list->elements;
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

void add_list_impl(struct List* list, size_t index, void* element) {
    size_t old_elements_count = list->elements;
    if (list->elements == list->space)
        grow_list(list);

    size_t element_size = list->elem_size;
    void* insert_at = (void*) ((size_t) list->alloc + element_size * index);
    void* dst = (void*) ((size_t) list->alloc + element_size * (index + 1));
    size_t amount = old_elements_count - index;
    memmove(dst, insert_at, element_size * amount);
    memcpy(insert_at, element, element_size);

    list->elements++;
}

void delete_list_impl(struct List* list, size_t index) {
    size_t old_elements_count = list->elements;

    size_t element_size = list->elem_size;

    void* hole_at = (void*) ((size_t) list->alloc + element_size * index);
    void* fill_with = (void*) ((size_t) list->alloc + element_size * (index + 1));
    size_t amount = old_elements_count - index;
    memmove(hole_at, fill_with, element_size * amount);

    list->elements--;
}

void* remove_list_impl(struct List* list, size_t index) {
    size_t old_elements_count = list->elements;

    size_t element_size = list->elem_size;
    char temp[element_size];

    void* hole_at = (void*) ((size_t) list->alloc + element_size * index);
    void* fill_with = (void*) ((size_t) list->alloc + element_size * (index + 1));
    size_t amount = old_elements_count - index;
    memcpy(&temp, hole_at, element_size);
    memmove(hole_at, fill_with, element_size * amount);

    list->elements--;

    void* end = (void*) ((size_t) list->alloc + element_size * list->elements);
    memcpy(end, &temp, element_size);
    return end;
}