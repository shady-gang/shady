#ifndef SHADY_LIST_H
#define SHADY_LIST_H

#include <stddef.h>
#include <stdbool.h>

struct List {
    size_t elements_count;
    size_t space;
    size_t element_size;
    void* alloc;
};

#define shd_new_list(T) shd_new_list_impl(sizeof(T))
struct List* shd_new_list_impl(size_t elem_size);

void shd_destroy_list(struct List* list);

size_t shd_list_count(struct List* list);

#define shd_list_append(T, list, element) shd_list_append_impl(list, (void*) &(element))
void shd_list_append_impl(struct List* list, void* element);

#define shd_list_pop(T, list) * ((T*) shd_list_pop_impl(list))
void* shd_list_pop_impl(struct List* list);

void shd_clear_list(struct List* list);

#define shd_list_insert(T, list, i, e) shd_list_insert_impl(list, i, (void*) &(e))
void shd_list_insert_impl(struct List* list, size_t index, void* element);

#define shd_list_delete(T, list, i) shd_list_remove_impl(list, index, false)
#define shd_list_remove(T, list, i) *((T*) shd_list_remove_impl(list, i, true))
void* shd_list_remove_impl(struct List* list, size_t index, bool);

#define shd_read_list(T, list) ((T*) (list)->alloc)

#endif
