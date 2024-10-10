#ifndef SHADY_DICT_H
#define SHADY_DICT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>

#include "portability.h"

typedef uint32_t KeyHash;
typedef KeyHash (*HashFn)(void*);
typedef bool (*CmpFn)(void*, void*);

struct Dict;

#define shd_new_dict(K, T, hash, cmp) shd_new_dict_impl(sizeof(K), sizeof(T), alignof(K), alignof(T), hash, cmp)
#define shd_new_set(K, hash, cmp) shd_new_dict_impl(sizeof(K), 0, alignof(K), 0, hash, cmp)
struct Dict* shd_new_dict_impl(size_t key_size, size_t value_size, size_t key_align, size_t value_align, KeyHash (* hash_fn)(void*), bool (* cmp_fn)(void*, void*));

struct Dict* shd_clone_dict(struct Dict* source);
void shd_destroy_dict(struct Dict* dict);
void shd_dict_clear(struct Dict* dict);

bool shd_dict_iter(struct Dict* dict, size_t* iterator_state, void* key, void* value);

size_t shd_dict_count(struct Dict* dict);

#define shd_dict_find_value(K, T, dict, key) (T*) shd_dict_find_value_impl(dict, (void*) (&(key)))
#define shd_dict_find_key(K, dict, key) (K*) shd_dict_find_impl(dict, (void*) (&(key)))
void* shd_dict_find_impl(struct Dict*, void*);
void* shd_dict_find_value_impl(struct Dict*, void*);

#define shd_dict_remove(K, dict, key) shd_dict_remove_impl(dict, (void*) (&(key)))
bool shd_dict_remove_impl(struct Dict* dict, void* key);

#define shd_dict_insert_get_value(K, V, dict, key, value) *(V*) shd_dict_insert_get_value_impl(dict, (void*) (&(key)), (void*) (&(value)))
#define shd_dict_insert(K, V, dict, key, value)                     shd_dict_insert_get_value_impl(dict, (void*) (&(key)), (void*) (&(value)))
void* shd_dict_insert_get_value_impl(struct Dict*, void* key, void* value);

#define shd_dict_insert_get_key(K, V, dict, key, value) *(K*) shd_dict_insert_get_key_impl(dict, (void*) (&(key)), (void*) (&(value)))
#define  shd_set_insert_get_key(K, dict, key)           *(K*) shd_dict_insert_get_key_impl(dict, (void*) (&(key)), NULL)
void* shd_dict_insert_get_key_impl(struct Dict*, void* key, void* value);

#define shd_dict_insert_get_result(K, V, dict, key, value) shd_dict_insert_impl(dict, (void*) (&(key)), (void*) (&(value)))
#define  shd_set_insert_get_result(K, dict, key)           shd_dict_insert_impl(dict, (void*) (&(key)), NULL)
bool shd_dict_insert_impl(struct Dict*, void* key, void* value);

KeyHash shd_hash(const void* data, size_t size);

KeyHash shd_hash_ptr(void**);
bool shd_compare_ptrs(void**, void**);

#endif
