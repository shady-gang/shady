#include "shady_meta_intrinsics.h"
#include "l2s_private.h"

#include "log.h"
#include "util.h"
#include "dict.h"

#include <stdlib.h>
#include <assert.h>

struct ShdIntrinsics_ {
    LLVMModuleRef module;
    Arena* arena;
    size_t ids_count;
    shady_parsed_meta_instruction** id_definitions;
    struct Dict* name_map;
};

static void scan_definition_for_id(ShdIntrinsics* intrinsics, String name, LLVMValueRef global) {
    global = LLVMGetInitializer(global);
    LLVMValueRef meta = LLVMGetAggregateElement(global, 0);
    uint64_t meta_value = LLVMConstIntGetZExtValue(meta);
    shady_meta_id defined_id = 0;
    switch ((shady_meta_instruction) meta_value) {
        case SHADY_META_INVALID: assert(false); break;
        case SHADY_META_DEFINE_LITERAL_I32:
        case SHADY_META_DEFINE_LITERAL_STRING:
        case SHADY_META_DEFINE_BUILTIN_TYPE:
        case SHADY_META_DEFINE_PARAM_REF:
        case SHADY_META_DEFINE_EXT_OP:
            defined_id = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 1));
            break;
        default:
            return;
    }
    assert(defined_id >= SHADY_META_IDS_BEGIN_AT);
    defined_id -= SHADY_META_IDS_BEGIN_AT;
    if (defined_id >= intrinsics->ids_count)
        intrinsics->ids_count = defined_id + 1;
    shd_dict_insert(String, shady_meta_id, intrinsics->name_map, name, defined_id);
    return;
}

static void parse_definition(ShdIntrinsics* intrinsics, LLVMValueRef global) {
    global = LLVMGetInitializer(global);
    LLVMValueRef meta = LLVMGetAggregateElement(global, 0);
    uint64_t meta_value = LLVMConstIntGetZExtValue(meta);
    uint64_t defined_id = UINT64_MAX;
    switch ((shady_meta_instruction) meta_value) {
        case SHADY_META_INVALID: assert(false); break;
        case SHADY_META_DEFINE_LITERAL_I32: {
            defined_id = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 1)) - SHADY_META_IDS_BEGIN_AT;
            uint32_t literal = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 2));
            shd_log_fmt(DEBUGV, "meta[%d] = literal_i32 %d\n", defined_id, literal);
            shady_parsed_meta_instruction* parsed = shd_arena_alloc(intrinsics->arena, sizeof(shady_parsed_meta_instruction));
            *parsed = (shady_parsed_meta_instruction) {
                .meta = meta_value,
                .literal_i32 = {
                    .defined_id = defined_id,
                    .literal = literal,
                }
            };
            intrinsics->id_definitions[defined_id] = (shady_parsed_meta_instruction*) parsed;
            return;
        } case SHADY_META_DEFINE_LITERAL_STRING: {
            defined_id = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 1)) - SHADY_META_IDS_BEGIN_AT;
            size_t len;
            const char* string = LLVMGetAsString(LLVMGetInitializer(LLVMGetAggregateElement(global, 2)), &len);
            shd_log_fmt(DEBUGV, "meta[%d] = literal_string '%s'\n", defined_id, string);
            shady_parsed_meta_instruction* parsed = shd_arena_alloc(intrinsics->arena, sizeof(shady_parsed_meta_instruction));
            char* allocated = shd_arena_alloc(intrinsics->arena, len + 1);
            memcpy(allocated, string, len + 1);
            *parsed = (shady_parsed_meta_instruction) {
                .meta = meta_value,
                .literal_string = {
                    .defined_id = defined_id,
                    .literal = allocated,
                }
            };
            intrinsics->id_definitions[defined_id] = (shady_parsed_meta_instruction*) parsed;
            return;
        }
        case SHADY_META_DEFINE_BUILTIN_TYPE: {
            defined_id = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 1)) - SHADY_META_IDS_BEGIN_AT;
            LLVMTypeRef t = LLVMTypeOf(LLVMGetAggregateElement(global, 2));
            shd_log_fmt(DEBUGV, "meta[%d] = type %zx\n", defined_id, t);
            shady_parsed_meta_instruction* parsed = shd_arena_alloc(intrinsics->arena, sizeof(shady_parsed_meta_instruction));
            *parsed = (shady_parsed_meta_instruction) {
                .meta = meta_value,
                .builtin_type = {
                    .defined_id = defined_id,
                    .type = t,
                }
            };
            intrinsics->id_definitions[defined_id] = (shady_parsed_meta_instruction*) parsed;
            return;
        } case SHADY_META_DEFINE_PARAM_REF: {
            defined_id = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 1)) - SHADY_META_IDS_BEGIN_AT;
            uint32_t idx = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 2));
            shd_log_fmt(DEBUGV, "meta[%d] = param_idx %d\n", defined_id, idx);
            shady_parsed_meta_instruction* parsed = shd_arena_alloc(intrinsics->arena, sizeof(shady_parsed_meta_instruction));
            *parsed = (shady_parsed_meta_instruction) {
                .meta = meta_value,
                .param_ref = {
                    .defined_id = defined_id,
                    .param_idx = idx,
                }
            };
            intrinsics->id_definitions[defined_id] = (shady_parsed_meta_instruction*) parsed;
            return;
        } case SHADY_META_DEFINE_EXT_OP: {
            defined_id = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 1)) - SHADY_META_IDS_BEGIN_AT;
            uint32_t op = LLVMConstIntGetZExtValue(LLVMGetAggregateElement(global, 2));
            LLVMValueRef ops = LLVMGetAggregateElement(global, 4);
            shady_meta_id* operands = NULL;
            size_t len;
            size_t num_operands = 0;
            if (ops) {
                char* src = LLVMGetAsString(LLVMGetInitializer(ops), &len);
                char* allocated = shd_arena_alloc(intrinsics->arena, len + 1);
                memcpy(allocated, src, len);
                operands = (shady_meta_id*) allocated;
                num_operands = len / 4;
            }
            shd_log_fmt(DEBUGV, "meta[%d] = ext_op %d(", defined_id, op);
            for (size_t i = 0; i < num_operands; i++) {
                operands[i] -= SHADY_META_IDS_BEGIN_AT;
                shd_log_fmt(DEBUGV, "%d", operands[i]);
                if (i + 1 < num_operands)
                    shd_log_fmt(DEBUGV, ", ");
            }
            shd_log_fmt(DEBUGV, ")\n", defined_id, op);
            shady_parsed_meta_instruction* parsed = shd_arena_alloc(intrinsics->arena, sizeof(shady_parsed_meta_instruction));
            *parsed = (shady_parsed_meta_instruction) {
                .meta = meta_value,
                .ext_op = {
                    .defined_id = defined_id,
                    .op_code = op,
                    .num_operands = num_operands,
                    .operands = operands,
                }
            };
            intrinsics->id_definitions[defined_id] = (shady_parsed_meta_instruction*) parsed;
            return;
        }
    }
}

KeyHash shd_hash_string(const char** string);
bool shd_compare_string(const char** a, const char** b);

ShdIntrinsics* create_shd_intrinsics(LLVMModuleRef mod) {
    ShdIntrinsics* intrinsics = calloc(sizeof(ShdIntrinsics), 1);
    *intrinsics = (ShdIntrinsics) {
        .module = mod,
        .arena = shd_new_arena(),
    };

    intrinsics->name_map = shd_new_dict(String, uint32_t, (HashFn) shd_hash_string, (CmpFn) shd_compare_string);

    LLVMValueRef global = LLVMGetFirstGlobal(mod);
    while (global) {
        const char* name = LLVMGetValueName(global);
        if (shd_string_starts_with(name, "__shady_meta_op_"))
            scan_definition_for_id(intrinsics, name + strlen("__shady_meta_op_"), global);
        if (global == LLVMGetLastGlobal(mod))
            break;
        global = LLVMGetNextGlobal(global);
    }

    shd_log_fmt(DEBUG, "Shady meta intrinsics parser: found %d meta ID definitions\n", intrinsics->ids_count);
    intrinsics->id_definitions = calloc(sizeof(shady_parsed_meta_instruction*), intrinsics->ids_count);

    global = LLVMGetFirstGlobal(mod);
    while (global) {
        const char* name = LLVMGetValueName(global);
        if (shd_string_starts_with(name, "__shady_meta_op_"))
            parse_definition(intrinsics, global);
        if (global == LLVMGetLastGlobal(mod))
            break;
        global = LLVMGetNextGlobal(global);
    }

    return intrinsics;
}

const shady_parsed_meta_instruction* shd_meta_id_definition(const ShdIntrinsics* intrinsics, size_t id) {
    assert(id < intrinsics->ids_count);
    return intrinsics->id_definitions[id];
}

shady_meta_id shd_meta_id_from_name(const ShdIntrinsics* intrinsics, const char* str) {
    shady_meta_id* found = shd_dict_find_value(String, uint32_t, intrinsics->name_map, str);
    if (found)
        return *found;
    return -1;
}

void destroy_shd_intrinsics(ShdIntrinsics* intrinsics) {
    shd_destroy_arena(intrinsics->arena);
    shd_destroy_dict(intrinsics->name_map);
    free(intrinsics->id_definitions);
    free(intrinsics);
}
