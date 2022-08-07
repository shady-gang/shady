#include "list.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "portability.h"

struct List* new_list_impl(size_t elem_size) {
    struct List* list = (struct List*) malloc(sizeof (struct List));
    *list = (struct List) {
        .elements_count = 0,
        .space = 8,
        .element_size = elem_size,
        .alloc = malloc(elem_size * 8)
    };
    return list;
}

void destroy_list(struct List* list) {
    free(list->alloc);
    free(list);
}

size_t entries_count_list(struct List* list) {
    return list->elements_count;
}

void grow_list(struct List* list) {
    list->space = list->space * 2;
    list->alloc = realloc(list->alloc, list->space * list->element_size);
}

void append_list_impl(struct List* list, void* element) {
    if (list->elements_count == list->space)
        grow_list(list);
    size_t element_size = list->element_size;
    memcpy((void*) ((size_t) list->alloc + element_size * list->elements_count), element, element_size);
    list->elements_count++;
}

void* pop_list_impl(struct List* list) {
    assert(list->elements_count > 0);
    list->elements_count--;
    void* last = (void*) ((size_t)(list->alloc) + list->elements_count * list->element_size);
    return last;
}

void add_list_impl(struct List* list, size_t index, void* element) {
    size_t old_elements_count = list->elements_count;
    if (list->elements_count == list->space)
        grow_list(list);

    size_t element_size = list->element_size;
    void* insert_at = (void*) ((size_t) list->alloc + element_size * index);
    void* dst = (void*) ((size_t) list->alloc + element_size * (index + 1));
    size_t amount = old_elements_count - index;
    memmove(dst, insert_at, element_size * amount);
    memcpy(insert_at, element, element_size);

    list->elements_count++;
}

void delete_list_impl(struct List* list, size_t index) {
    size_t old_elements_count = list->elements_count;

    size_t element_size = list->element_size;

    void* hole_at = (void*) ((size_t) list->alloc + element_size * index);
    void* fill_with = (void*) ((size_t) list->alloc + element_size * (index + 1));
    size_t amount = old_elements_count - index;
    memmove(hole_at, fill_with, element_size * amount);

    list->elements_count--;
}

void* remove_list_impl(struct List* list, size_t index) {
    size_t old_elements_count = list->elements_count;

    size_t element_size = list->element_size;
    LARRAY(char, temp, element_size);

    void* hole_at = (void*) ((size_t) list->alloc + element_size * index);
    void* fill_with = (void*) ((size_t) list->alloc + element_size * (index + 1));
    size_t amount = old_elements_count - index;
    memcpy(&temp, hole_at, element_size);
    memmove(hole_at, fill_with, element_size * amount);

    list->elements_count--;

    void* end = (void*) ((size_t) list->alloc + element_size * list->elements_count);
    memcpy(end, &temp, element_size);
    return end;
}

void clear_list(struct List* list) {
    list->elements_count = 0;
}
