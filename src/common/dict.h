#ifndef SHADY_DICT_H
#define SHADY_DICT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>

typedef uint32_t KeyHash;
typedef KeyHash (*HashFn)(void*);
typedef bool (*CmpFn)(void*, void*);

struct Dict;

#define new_dict(K, T, hash, cmp) new_dict_impl(sizeof(K), sizeof(T), alignof(K), alignof(T), hash, cmp)
#define new_set(K, hash, cmp) new_dict_impl(sizeof(K), 0, alignof(K), 0, hash, cmp)
struct Dict* new_dict_impl(size_t key_size, size_t value_size, size_t key_align, size_t value_align, KeyHash (*)(void*), bool (*)(void*, void*));

struct Dict* clone_dict(struct Dict*);
void destroy_dict(struct Dict*);
void clear_dict(struct Dict*);

bool dict_iter(struct Dict*, size_t* iterator_state, void* key, void* value);

size_t entries_count_dict(struct Dict*);

#define find_value_dict(K, T, dict, key) (T*) find_value_dict_impl(dict, (void*) (&(key)))
#define find_key_dict(K, dict, key) (K*) find_key_dict_impl(dict, (void*) (&(key)))
void* find_key_dict_impl(struct Dict*, void*);
void* find_value_dict_impl(struct Dict*, void*);

#define remove_dict(K, dict, key) remove_dict_impl(dict, (void*) (&(key)))
bool remove_dict_impl(struct Dict* dict, void* key);

#define insert_dict_and_get_value(K, V, dict, key, value) *(V*) insert_dict_and_get_value_impl(dict, (void*) (&(key)), (void*) (&(value)))
#define insert_dict(K, V, dict, key, value)                     insert_dict_and_get_value_impl(dict, (void*) (&(key)), (void*) (&(value)))
void* insert_dict_and_get_value_impl(struct Dict*, void* key, void* value);

#define insert_dict_and_get_key(K, V, dict, key, value) *(K*) insert_dict_and_get_key_impl(dict, (void*) (&(key)), (void*) (&(value)))
#define      insert_set_get_key(K, dict, key)           *(K*) insert_dict_and_get_key_impl(dict, (void*) (&(key)), NULL)
void* insert_dict_and_get_key_impl(struct Dict*, void* key, void* value);

#define insert_dict_and_get_result(K, V, dict, key, value) insert_dict_and_get_result_impl(dict, (void*) (&(key)), (void*) (&(value)))
#define      insert_set_get_result(K, dict, key)           insert_dict_and_get_result_impl(dict, (void*) (&(key)), NULL)
bool insert_dict_and_get_result_impl(struct Dict*, void* key, void* value);

KeyHash hash_murmur(const void* data, size_t size);

KeyHash hash_ptr(void**);
bool compare_ptrs(void**, void**);

#endif
