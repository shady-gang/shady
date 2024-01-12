#include <json-c/json.h>
#include <assert.h>

#include "log.h"

void json_apply_object(json_object* target, json_object* src);

void json_apply_object(json_object* target, json_object* src) {
    assert(target && src);
    assert(json_object_get_type(target) == json_type_object);
    assert(json_object_get_type(src) == json_type_object);
    struct json_object_iterator end = json_object_iter_end(src);
    for (struct json_object_iterator i = json_object_iter_begin(src); !json_object_iter_equal(&end, &i); json_object_iter_next(&i)) {
        const char* name = json_object_iter_peek_name(&i);
        json_object* value = json_object_iter_peek_value(&i);

        json_object* existing = json_object_object_get(target, name);
        if (existing && json_object_get_type(existing) == json_type_object) {
            json_apply_object(existing, value);
        } else if (existing && json_object_get_type(existing) == json_type_array && json_object_get_type(value) == json_type_array && json_object_array_length(value) <= json_object_array_length(existing)) {
            for (size_t j = 0; j < json_object_array_length(value); j++)
                json_object_array_put_idx(existing, j, json_object_array_get_idx(value, j));
        } else {
            if (existing)
                warn_print("json-apply: overwriting key '%s'\n", name);
            json_object_object_add(target, name, value);
            json_object_get(value);
        }
    }
}