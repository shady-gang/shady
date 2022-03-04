#ifndef SHADY_DICT_H
#define SHADY_DICT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>

typedef uint32_t KeyHash;

struct Dict;

#define new_dict(K, T, hash, cmp) new_dict_impl(sizeof(K), sizeof(T), alignof(K), alignof(T), hash, cmp)
#define new_set(K, hash, cmp) new_dict_impl(sizeof(K), 0, alignof(K), 0, hash, cmp)
struct Dict* new_dict_impl(size_t key_size, size_t value_size, size_t key_align, size_t value_align, KeyHash (*)(void*), bool (*)(void*, void*));

void destroy_dict(struct Dict*);

size_t entries_count_dict(struct Dict*);

#define find_value_dict(K, T, dict, key) (T*) find_value_dict_impl(dict, (void*) (&(key))
#define find_key_dict(K, dict, key) (K*) find_key_dict_impl(dict, (void*) (&(key)))
void* find_key_dict_impl(struct Dict*, void*);
void* find_value_dict_impl(struct Dict*, void*);

#define insert_dict(K, V, dict, key, value) insert_dict_impl_no_out_ptr(dict, (void*) (&(key)), &(void*) (&(value)))
bool insert_dict_impl(struct Dict*, void* key, void* value, void** out_ptr);

#define insert_or_get_set(K, dict, key) (K*) insert_dict_impl_and_get_key(dict, (void*) (&(key)), NULL)
void* insert_dict_impl_and_get_value(struct Dict*, void* key, void* value);
void* insert_dict_impl_and_get_key(struct Dict*, void* key, void* value);
bool insert_dict_impl_and_get_result(struct Dict*, void* key, void* value);

#endif
