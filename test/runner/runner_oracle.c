#include "runner_app_common.h"
#include "shady/runtime/runtime.h"

#include "json.h"

#include "portability.h"

const Node* parse_pattern(IrArena* a, const Type* datatype, json_object* pattern) {
    switch (json_object_get_type(pattern)) {
        case json_type_array: {
            uint32_t length = json_object_array_length(pattern);
            LARRAY(const Node*, arr, length);
            for (size_t i = 0; i < length; i++) {
                arr[i] = parse_pattern(a, datatype, json_object_array_get_idx(pattern, i));
            }
            return composite_helper(a, arr_type_helper(a, 0, datatype, shd_uint32_literal(a, length)), shd_nodes(a, length, arr));
        }
        case json_type_null:break;
        case json_type_boolean:break;
        case json_type_double:break;
        case json_type_int: {
            assert(datatype->tag == Int_TAG);
            Int int_type = datatype->payload.int_type;
            return int_literal_helper(a, int_type.width, int_type.is_signed, json_object_get_int(pattern));
        }
        case json_type_object:break;
        case json_type_string:break;
    }
    shd_error("Unhandled pattern");
}

void parse_arg_config(const ShdRunnerOracleConfig* config, IrArena* a, json_object* arg_config, ShdRunnerOracleArg* arg) {
    String kind = json_object_get_string(json_object_object_get(arg_config, "kind"));
    if (strcmp(kind, "value") == 0)
        arg->kind = ShdRunnerOracleArg_kind_VALUE;
    else if (strcmp(kind, "buffer") == 0)
        arg->kind = ShdRunnerOracleArg_kind_BUFFER;
    else shd_error("Unknown argument kind");

    String datatype = json_object_get_string(json_object_object_get(arg_config, "datatype"));
    if (strcmp(datatype, "i32") == 0)
        arg->type = shd_int32_type(a);
    else shd_error("Unknown datatype");

    if (arg->kind == ShdRunnerOracleArg_kind_BUFFER) {
        json_object* per_invocation_size = json_object_object_get(arg_config, "per-invocation-size");
        if (per_invocation_size) {
            arg->buffer_size = config->dispatch_size[0] * config->dispatch_size[1] * config->dispatch_size[2] * json_object_get_int(per_invocation_size);
        }
        assert(arg->buffer_size > 0);

        json_object* pre_pattern = json_object_object_get(arg_config, "pre-pattern");
        if (pre_pattern)
            arg->pre_pattern = parse_pattern(a, arg->type, pre_pattern);

        json_object* post_pattern = json_object_object_get(arg_config, "post-pattern");
        if (post_pattern)
            arg->post_pattern = parse_pattern(a, arg->type, post_pattern);
    } else {
        json_object* value = json_object_object_get(arg_config, "value");
        if (value)
            arg->value = parse_pattern(a, arg->type, value);
    }
}

ShdRunnerOracleConfig shd_runner_oracle_parse_config(IrArena* a, String json) {
    ShdRunnerOracleConfig config = { 0 };
    json_tokener* tokener = json_tokener_new();
    json_object* root = json_tokener_parse(json);

    json_object* dispatch_size = json_object_object_get(root, "dispatch-size");
    config.dispatch_size[0] = json_object_get_int(json_object_array_get_idx(dispatch_size, 0));
    assert(config.dispatch_size[0] > 0);
    config.dispatch_size[1] = json_object_get_int(json_object_array_get_idx(dispatch_size, 1));
    assert(config.dispatch_size[1] > 0);
    config.dispatch_size[2] = json_object_get_int(json_object_array_get_idx(dispatch_size, 2));
    assert(config.dispatch_size[2] > 0);

    json_object* args = json_object_object_get(root, "dispatch-args");
    config.num_args = json_object_array_length(args);
    config.args = calloc(sizeof(ShdRunnerOracleArg), config.num_args);

    for (size_t i = 0; i < config.num_args; i++) {
        parse_arg_config(&config, a, json_object_array_get_idx(args, i), &config.args[i]);
    }

    json_object_put(root);
    json_tokener_free(tokener);

    return config;
}

void shd_runner_oracle_free_config(ShdRunnerOracleConfig* config) {
    free(config->args);
}

void shd_runner_oracle_prefill(void* dst, const size_t size, const Node* pattern) {
    const size_t pattern_size = shd_rt_get_size_of_constant(pattern);
    shd_rt_materialize_constant_at(dst, pattern);
    size_t offset = pattern_size;
    while (offset < size) {
        memcpy((void*) ((size_t) dst + offset), dst, pattern_size);
        offset += pattern_size;
    }
}

bool shd_runner_oracle_validate(void* dst, size_t size, const Node* pattern) {
    size_t pattern_size = shd_rt_get_size_of_constant(pattern);
    void* reference = malloc(pattern_size);
    shd_rt_materialize_constant_at(reference, pattern);
    size_t offset = 0;
    while (offset < size) {
        for (size_t i = 0; i < pattern_size; i++) {
            char ref = ((char*)reference)[i];
            char got = ((char*)dst)[offset + i];
            if (got != ref) {
                shd_log_fmt(ERROR, "Validation of output failed: got 0x%x instead of 0x%x at byte offset 0x%x.", got, ref, offset + i);
                return false;
            }
        }
        offset += pattern_size;
    }
    free(reference);
    return true;
}