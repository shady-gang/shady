#include "dict.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

inline static size_t div_roundup(size_t a, size_t b) {
    if (a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

inline static size_t align_offset(size_t offset, size_t alignment) {
    if (!alignment) return offset;
    return div_roundup(offset, alignment) * alignment;
}

inline static size_t maxof(size_t a, size_t b) {
    return a > b ? a : b;
}

static size_t init_size = 32;

struct BucketTag {
    bool is_present;
    bool is_thombstone;
};

struct Dict {
    size_t entries_count;
    size_t size;

    size_t key_size;
    size_t value_size;

    size_t value_offset;
    size_t tag_offset;
    size_t bucket_entry_size;

    KeyHash (*hash_fn) (void*);
    bool (*cmp_fn) (void*, void*);
    void* alloc;
};

struct Dict* new_dict_impl(size_t key_size, size_t value_size, size_t key_align, size_t value_align, KeyHash (*hash_fn)(void*), bool (*cmp_fn) (void*, void*)) {
    // offset of key is obviously zero
    size_t value_offset = align_offset(key_size, value_align);
    size_t tag_offset = align_offset(value_offset + value_size, alignof(struct BucketTag));

    size_t bucket_entry_size = tag_offset + sizeof(struct BucketTag);

    // Add extra padding at the end of each entry if required...
    size_t max_align = maxof(maxof(key_align, value_align), alignof(struct BucketTag));
    bucket_entry_size = align_offset(bucket_entry_size, max_align);

    struct Dict* dict = (struct Dict*) malloc(sizeof(struct Dict));
    *dict = (struct Dict) {
        .entries_count = 0,
        .size = init_size,

        .key_size = key_size,
        .value_size = value_size,

        .value_offset = value_offset,
        .tag_offset = tag_offset,
        .bucket_entry_size = bucket_entry_size,

        .hash_fn = hash_fn,
        .cmp_fn = cmp_fn,

        .alloc = malloc(bucket_entry_size * init_size)
    };
    // zero-init
    memset(dict->alloc, 0, bucket_entry_size * init_size);
    return dict;
}

struct Dict* clone_dict(struct Dict* source) {
    struct Dict* dict = (struct Dict*) malloc(sizeof(struct Dict));
    *dict = (struct Dict) {
        .entries_count = source->entries_count,
        .size = source->size,

        .key_size = source->key_size,
        .value_size = source->value_size,

        .value_offset = source->value_offset,
        .tag_offset = source->tag_offset,
        .bucket_entry_size = source->bucket_entry_size,

        .hash_fn = source->hash_fn,
        .cmp_fn = source->cmp_fn,

        .alloc = malloc(source->bucket_entry_size * source->size)
    };
    memcpy(dict->alloc, source->alloc, source->bucket_entry_size * source->size);
    return dict;
}

void destroy_dict(struct Dict* dict) {
    free(dict->alloc);
    free(dict);
}

size_t entries_count_dict(struct Dict* dict) {
    return dict->entries_count;
}

void* find_key_dict_impl(struct Dict* dict, void* key) {
    KeyHash hash = dict->hash_fn(key);
    size_t pos = hash % dict->size;
    const size_t init_pos = pos;
    const size_t alloc_base = (size_t) dict->alloc;
    while (true) {
        size_t bucket = alloc_base + pos * dict->bucket_entry_size;

        void* in_dict_key = (void*) bucket;
        struct BucketTag* tag = (struct BucketTag*) (void*) (bucket + dict->tag_offset);
        if (tag->is_present || tag->is_thombstone) {
            // If the key is identical, we found our guy !
            if (tag->is_present && dict->cmp_fn(in_dict_key, key))
                return in_dict_key;

            // Otherwise, do a crappy linear scan...
            pos++;
            if (pos == dict->size)
                pos = 0;

            // Make sure to die if we go full circle
            if (pos == init_pos)
                break;
        } else break;
    }
    return NULL;
}

void* find_value_dict_impl(struct Dict* dict, void* key) {
    void* found = find_key_dict_impl(dict, key);
    if (found)
        return (void*) ((size_t)found + dict->value_offset);
    return NULL;
}

bool remove_dict_impl(struct Dict* dict, void* key) {
    void* found = find_key_dict_impl(dict, key);
    if (found) {
        struct BucketTag* tag = (void *) ((size_t) found + dict->tag_offset);
        tag->is_present = false;
        tag->is_thombstone = true;
        return true;
    }
    return false;
}

bool insert_dict_impl(struct Dict* dict, void* key, void* value, void** out_ptr);
bool insert_dict_and_get_result_impl(struct Dict* dict, void* key, void* value) {
    void* dont_care;
    return insert_dict_impl(dict, key, value, &dont_care);
}

void* insert_dict_and_get_key_impl(struct Dict* dict, void* key, void* value) {
    void* do_care;
    insert_dict_impl(dict, key, value, &do_care);
    return do_care;
}

void* insert_dict_and_get_value_impl(struct Dict* dict, void* key, void* value) {
    void* do_care;
    insert_dict_impl(dict, key, value, &do_care);
    return (void*) ((size_t)do_care + dict->value_offset);
}

void rehash(struct Dict* dict, void* old_alloc, size_t old_size) {
    const size_t alloc_base = (size_t) old_alloc;
    // Go over all the old entries and add them back
    for(size_t pos = 0; pos < old_size; pos++) {
        size_t bucket = alloc_base + pos * dict->bucket_entry_size;

        struct BucketTag* tag = (struct BucketTag*) (void*) (bucket + dict->tag_offset);
        if (tag->is_present) {
            void* key = (void*) bucket;
            void* value = (void*) (bucket + dict->value_offset);
            insert_dict_and_get_result_impl(dict, key, value);
        }
    }
}

void grow_and_rehash(struct Dict* dict) {
    printf("grow_rehash");

    void* old_alloc = dict->alloc;
    size_t old_size = dict->size;

    dict->entries_count = 0;
    dict->size *= 2;
    dict->alloc = malloc(dict->size * dict->bucket_entry_size);

    rehash(dict, old_alloc, old_size);

    free(old_alloc);
}

bool insert_dict_impl(struct Dict* dict, void* key, void* value, void** out_ptr) {
    void* existing = find_key_dict_impl(dict, key);
    if (existing) {
        *out_ptr = existing;
        return false;
    }

    float load_factor = (float) dict->entries_count / (float) dict->size;
    if (load_factor > 0.6)
        grow_and_rehash(dict);

    KeyHash hash = dict->hash_fn(key);
    size_t pos = hash % dict->size;
    const size_t init_pos = pos;
    const size_t alloc_base = (size_t) dict->alloc;

    // Find an empty spot...
    while (true) {
        size_t bucket = alloc_base + pos * dict->bucket_entry_size;

        struct BucketTag tag = *(struct BucketTag*) (void*) (bucket + dict->tag_offset);
        if (tag.is_present) {
            pos++;

            if (pos == dict->size)
                pos = 0;
            // Make sure to die if we go full circle
            assert(pos != init_pos);
        } else break;
    }
    assert(pos < dict->size);

    size_t bucket = alloc_base + pos * dict->bucket_entry_size;
    struct BucketTag* tag = (struct BucketTag*) (void*) (bucket + dict->tag_offset);
    void* in_dict_key = (void*) bucket;
    void* in_dict_value = (void*) (bucket + dict->value_offset);
    assert(!tag->is_present);

    tag->is_present = true;
    tag->is_thombstone = false;
    memcpy(in_dict_key,   key,   dict->key_size);
    if (dict->value_size)
        memcpy(in_dict_value, value, dict->value_size);
    *out_ptr = in_dict_key;

    return true;
}
