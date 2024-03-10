#include "dict.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

inline static size_t div_roundup(size_t a, size_t b) {
    //return (a + b - 1) / b;
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
    size_t thombstones_count;
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
        .thombstones_count = 0,
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
        .thombstones_count = source->thombstones_count,
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

void clear_dict(struct Dict* dict) {
    dict->entries_count = 0;
    dict->thombstones_count = 0;
    memset(dict->alloc, 0, dict->bucket_entry_size * dict->size);
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
        struct BucketTag* tag = (void *) (((size_t) found) + dict->tag_offset);
        assert(tag->is_present && !tag->is_thombstone);
        tag->is_present = false;
        tag->is_thombstone = true;
        dict->thombstones_count++;
        dict->entries_count--;
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

static void rehash(struct Dict* dict, void* old_alloc, size_t old_size) {
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

static void grow_and_rehash(struct Dict* dict) {
    //printf("grow_rehash\n");
    size_t old_entries_count = entries_count_dict(dict);

    void* old_alloc = dict->alloc;
    size_t old_size = dict->size;

    dict->entries_count = 0;
    dict->thombstones_count = 0;
    dict->size *= 2;
    dict->alloc = malloc(dict->size * dict->bucket_entry_size);
    // zero-allocated so all the bucket flags are false
    memset(dict->alloc, 0, dict->size * dict->bucket_entry_size);

    rehash(dict, old_alloc, old_size);
    assert(old_entries_count == entries_count_dict(dict));

    free(old_alloc);
}

bool insert_dict_impl(struct Dict* dict, void* key, void* value, void** out_ptr) {
    float load_factor = (float) (dict->entries_count + dict->thombstones_count) / (float) dict->size;
    if (load_factor > 0.6)
        grow_and_rehash(dict);

    KeyHash hash = dict->hash_fn(key);
    size_t pos = hash % dict->size;
    const size_t init_pos = pos;
    const size_t alloc_base = (size_t) dict->alloc;

    enum { Inserting, Overwriting, Moving } mode;

    size_t first_available_pos = SIZE_MAX;

    // Find an empty spot...
    while (true) {
        size_t bucket = alloc_base + pos * dict->bucket_entry_size;

        struct BucketTag tag = *(struct BucketTag*) (void*) (bucket + dict->tag_offset);
        if (!tag.is_present) {
            if (first_available_pos == SIZE_MAX)
                first_available_pos = pos;
            if (!tag.is_thombstone) {
                mode = Inserting;
                break;
            }
        } else {
            void* in_dict_key = (void*) bucket;
            if (dict->cmp_fn(in_dict_key, key)) {
                if (first_available_pos == SIZE_MAX)
                    first_available_pos = pos;
                mode = (first_available_pos == pos) ? Overwriting : Moving;
                break;
            }
        }

        pos++;
        if (pos == dict->size)
            pos = 0;
        if (pos == init_pos) {
            assert(first_available_pos != SIZE_MAX);
            mode = Inserting;
            break;
        }
    }
    assert(first_available_pos < dict->size);
    assert(pos < dict->size);

    size_t dst_bucket = alloc_base + first_available_pos * dict->bucket_entry_size;
    struct BucketTag* dst_tag = (struct BucketTag*) (void*) (dst_bucket + dict->tag_offset);
    void* in_dict_key = (void*) dst_bucket;
    void* in_dict_value = (void*) (dst_bucket + dict->value_offset);

    if (dst_tag->is_thombstone)
        dict->thombstones_count--;

    if (mode == Moving) {
        size_t src_bucket = alloc_base + pos * dict->bucket_entry_size;
        struct BucketTag* src_tag = (struct BucketTag*) (void*) (src_bucket + dict->tag_offset);
        assert(src_tag->is_present && dst_tag->is_thombstone);
        src_tag->is_thombstone = true;
        src_tag->is_present = false;
    } else if (mode == Overwriting) {
        assert(dst_tag->is_present);
    } else {
        dict->entries_count++;
    }

    dst_tag->is_present = true;
    dst_tag->is_thombstone = false;
    memcpy(in_dict_key, key, dict->key_size);
    if (dict->value_size)
        memcpy(in_dict_value, value, dict->value_size);
    *out_ptr = in_dict_key;

    return mode == Inserting;
}

bool dict_iter(struct Dict* dict, size_t* iterator_state, void* key, void* value) {
    bool found_something = false;
    while (!found_something) {
        if (*iterator_state >= dict->size) {
            return false;
        }
        const size_t alloc_base = (size_t) dict->alloc;
        size_t bucket = alloc_base + (*iterator_state) * dict->bucket_entry_size;
        struct BucketTag* tag = (struct BucketTag*) (void*) (bucket + dict->tag_offset);
        if (tag->is_present) {
            found_something = true;
            void* in_dict_key = (void*) bucket;
            if (key)
                memcpy(key, in_dict_key, dict->key_size);
            void* in_dict_value = (void*) (bucket + dict->value_offset);
            if (value && dict->value_size > 0)
                memcpy(value, in_dict_value, dict->value_size);
        }
        (*iterator_state)++;
    }
    return true;
}

#include "murmur3.h"

KeyHash hash_murmur(const void* data, size_t size) {
    int32_t out[4];
    MurmurHash3_x64_128(data, (int) size, 0x1234567, &out);

    uint32_t final = 0;
    final ^= out[0];
    final ^= out[1];
    final ^= out[2];
    final ^= out[3];
    return final;
}

KeyHash hash_ptr(void** pptr) {
    return hash_murmur(pptr, sizeof(void*));
}

bool compare_ptr(void** pa, void** pb) {
    if (pa && pb) {
        return *pa == *pb;
    }
    return pa == pb;
}