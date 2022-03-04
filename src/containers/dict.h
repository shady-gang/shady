#ifndef SHADY_DICT_H
#define SHADY_DICT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>

typedef uint32_t KeyHash;

struct Dict;

#define new_dict(K, T, fn) new_dict_impl(sizeof(K), sizeof(T), alignof(K), alignof(T), fn)
struct Dict* new_dict_impl(size_t key_size, size_t value_size, size_t key_align, size_t value_align, KeyHash (*)(void*));

void destroy_dict(struct Dict*);

size_t entries_count_dict(struct Dict*);

#define find_dict(K, T, dict, key) (T*) find_dict_impl(dict, key)
void* find_dict_impl(struct Dict*, void*);

#define insert_dict(K, V, dict, key, value) insert_dict_impl_no_out_ptr(dict, key, value)
bool insert_dict_impl_no_out_ptr(struct Dict*, void* key, void* value);
bool insert_dict_impl(struct Dict*, void* key, void* value, void** out_ptr);

#endif
