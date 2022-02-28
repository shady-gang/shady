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
struct List* new_list_impl(size_t elem_size);

void destroy_list(struct List* list);

#define append_list(T, list, element) append_list_impl(list, (void*) &(element))
void append_list_impl(struct List* list, void* element);

#define pop_list(T, list) * ((T*) pop_list_impl(list))
void* pop_list_impl(struct List* list);

#define add_list(T, list, i, e) add_list_impl(list, i, (void*) &(e))
void add_list_impl(struct List* list, size_t index, void* element);

#define delete_list(T, list, i) delete_list_impl(list, index)
void delete_list_impl(struct List* list, size_t index);

#define remove_list(T, list, i) *(T*) remove_list_impl(list, index)
void* remove_list_impl(struct List* list, size_t index);

#endif
