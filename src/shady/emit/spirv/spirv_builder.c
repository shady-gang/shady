#include "spirv_builder.h"

#include "list.h"
#include "growy.h"
#include "dict.h"

#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#define SHADY_GENERATOR_MAGIC_NUMBER 35

inline static int div_roundup(int a, int b) {
    if (a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

inline static void output_word(SpvbSectionBuilder data, uint32_t word) {
    growy_append_bytes(data, sizeof(uint32_t), (char*) &word);
}

#define op(opcode, size) op_(target_data, opcode, size)
inline static void op_(SpvbSectionBuilder data, SpvOp op, int ops_size) {
    uint32_t lower = op & 0xFFFFu;
    uint32_t upper = (ops_size << 16) & 0xFFFF0000u;
    output_word(data, lower | upper);
}

#define ref_id(i) ref_id_(target_data, i)
inline static void ref_id_(SpvbSectionBuilder data, SpvId id) {
    assert(id != 0);
    output_word(data, id);
}

#define literal_name(str) spvb_literal_name(target_data, str)
void spvb_literal_name(SpvbSectionBuilder data, const char* str) {
    int i = 0;
    uint32_t cword = 0;
    while (str[0] != '\0') {
        char c = str[0];
        str = &str[1];
        cword = cword | (c & 0xFF) << (i * 8);
        i++;
        if (i == 4) {
            output_word(data, cword);
            cword = 0;
            i = 0;
        }
    }
    output_word(data, cword);
}

#define literal_int(i) literal_int_(target_data, i)
inline static void literal_int_(SpvbSectionBuilder data, uint32_t i) {
    output_word(data, i);
}

#define copy_section(section) copy_section_(target_data, section)
inline static void copy_section_(SpvbSectionBuilder target, SpvbSectionBuilder source) {
    growy_append_bytes(target, growy_size(source), growy_data(source));
}

struct SpvbFileBuilder_ {
    SpvAddressingModel addressing_model;
    SpvMemoryModel memory_model;

    uint32_t bound;

    struct {
        uint8_t major;
        uint8_t minor;
    } version;

    // Ordered as per https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.pdf#subsection.2.4
    SpvbSectionBuilder capabilities;
    SpvbSectionBuilder extensions;
    SpvbSectionBuilder ext_inst_import;
    SpvbSectionBuilder entry_points;
    SpvbSectionBuilder execution_modes;
    SpvbSectionBuilder debug_string_source;
    SpvbSectionBuilder debug_names;
    SpvbSectionBuilder debug_module_processed;
    SpvbSectionBuilder annotations;
    SpvbSectionBuilder types_constants;
    SpvbSectionBuilder fn_decls;
    SpvbSectionBuilder fn_defs;

    struct Dict* capabilities_set;
    struct Dict* extensions_set;
};

static KeyHash hash_u32(uint32_t* p) { return hash_murmur(p, sizeof(uint32_t)); }
static bool compare_u32s(uint32_t* a, uint32_t* b) { return *a == *b; }

KeyHash hash_string(const char** string);
bool compare_string(const char** a, const char** b);

SpvbFileBuilder* spvb_begin() {
    SpvbFileBuilder* file_builder = (SpvbFileBuilder*) malloc(sizeof(SpvbFileBuilder));
    *file_builder = (SpvbFileBuilder) {
        .bound = 1,
        .capabilities = new_growy(),
        .extensions = new_growy(),
        .ext_inst_import = new_growy(),
        .entry_points = new_growy(),
        .execution_modes = new_growy(),
        .debug_string_source = new_growy(),
        .debug_names = new_growy(),
        .debug_module_processed = new_growy(),
        .annotations = new_growy(),
        .types_constants = new_growy(),
        .fn_decls = new_growy(),
        .fn_defs = new_growy(),

        .capabilities_set = new_set(SpvCapability, (HashFn) hash_u32, (CmpFn) compare_u32s),
        .extensions_set = new_set(const char*, (HashFn) hash_string, (CmpFn) compare_string),

        .memory_model = SpvMemoryModelGLSL450,
    };
    return file_builder;
}

#define target_data final_output
inline static void merge_sections(SpvbFileBuilder* file_builder, SpvbSectionBuilder final_output) {
    literal_int(SpvMagicNumber);
    uint32_t version_tag = 0;
    version_tag |= ((uint32_t) file_builder->version.major) << 16;
    version_tag |= ((uint32_t) file_builder->version.minor) << 8;
    literal_int(version_tag);
    literal_int(SHADY_GENERATOR_MAGIC_NUMBER);
    literal_int(file_builder->bound);
    literal_int(0); // instruction schema padding

    copy_section(file_builder->capabilities);
    copy_section(file_builder->extensions);
    copy_section(file_builder->ext_inst_import);

    op(SpvOpMemoryModel, 3);
    literal_int(file_builder->addressing_model);
    literal_int(file_builder->memory_model);

    copy_section(file_builder->entry_points);
    copy_section(file_builder->execution_modes);
    copy_section(file_builder->debug_string_source);
    copy_section(file_builder->debug_names);
    copy_section(file_builder->debug_module_processed);
    copy_section(file_builder->annotations);
    copy_section(file_builder->types_constants);
    copy_section(file_builder->fn_decls);
    copy_section(file_builder->fn_defs);
}
#undef target_data

static const uint8_t endian_check_helper[4] = { 1, 2, 3, 4 };

static bool is_big_endian() {
    uint32_t x;
    memcpy(&x, &endian_check_helper, sizeof(x));
    if (x == 0x01020304)
        return true;
    assert(x == 0x04030201);
    return false;
}

static uint32_t byteswap(uint32_t v) {
    uint8_t arr[4];
    memcpy(&arr, &v, sizeof(v));
    uint8_t swizzled[4] = { arr[3], arr[2], arr[1], arr[0] };
    memcpy(&v, &swizzled, sizeof(v));
    return v;
}

size_t spvb_finish(SpvbFileBuilder* file_builder, char** output) {
    Growy* g = new_growy();
    merge_sections(file_builder, g);

    destroy_growy(file_builder->fn_defs);
    destroy_growy(file_builder->fn_decls);
    destroy_growy(file_builder->types_constants);
    destroy_growy(file_builder->annotations);
    destroy_growy(file_builder->debug_module_processed);
    destroy_growy(file_builder->debug_names);
    destroy_growy(file_builder->debug_string_source);
    destroy_growy(file_builder->execution_modes);
    destroy_growy(file_builder->entry_points);
    destroy_growy(file_builder->ext_inst_import);
    destroy_growy(file_builder->extensions);
    destroy_growy(file_builder->capabilities);

    destroy_dict(file_builder->capabilities_set);
    destroy_dict(file_builder->extensions_set);

    free(file_builder);

    size_t s = growy_size(g);
    assert(s % 4 == 0);
    *output = growy_deconstruct(g);

    if (is_big_endian()) for (size_t i = 0; i < s / 4; i++) {
        ((uint32_t*)*output)[i] = byteswap(((uint32_t*)(*output))[i]);
    }

    return s;
}

SpvId spvb_fresh_id(SpvbFileBuilder* file_builder) {
    return file_builder->bound++;
}

void spvb_set_version(SpvbFileBuilder* file_builder, uint8_t major, uint8_t minor) {
    file_builder->version.major = major;
    file_builder->version.minor = minor;
}

void spvb_set_addressing_model(SpvbFileBuilder* file_builder, SpvAddressingModel model) {
    assert(file_builder->addressing_model == SpvAddressingModelLogical || file_builder->addressing_model == model);
    file_builder->addressing_model = model;
}

#define target_data file_builder->capabilities
void spvb_capability(SpvbFileBuilder* file_builder, SpvCapability cap) {
    if (insert_set_get_result(SpvCapability, file_builder->capabilities_set, cap)) {
        op(SpvOpCapability, 2);
        literal_int(cap);
    }
}
#undef target_data

#define target_data file_builder->extensions
void spvb_extension(SpvbFileBuilder* file_builder, const char* name) {
    if (insert_set_get_result(char*, file_builder->extensions_set, name)) {
        op(SpvOpExtension, 1 + div_roundup(strlen(name) + 1, 4));
        literal_name(name);
    }
}
#undef target_data

#define target_data file_builder->ext_inst_import
SpvId spvb_extended_import(SpvbFileBuilder* file_builder, const char* name) {
    op(SpvOpExtInstImport, 2 + div_roundup(strlen(name) + 1, 4));
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_name(name);
    return id;
}
#undef target_data

#define target_data file_builder->entry_points
void spvb_entry_point(SpvbFileBuilder* file_builder, SpvExecutionModel execution_model, SpvId entry_point, const char* name, size_t interface_elements_count, SpvId interface_elements[]) {
    op(SpvOpEntryPoint, 3 + div_roundup(strlen(name) + 1, 4) + interface_elements_count);
    literal_int(execution_model);
    ref_id(entry_point);
    literal_name(name);
    for (size_t i = 0; i < interface_elements_count; i++)
        ref_id(interface_elements[i]);
}
#undef target_data

#define target_data file_builder->execution_modes
void spvb_execution_mode(SpvbFileBuilder* file_builder, SpvId entry_point, SpvExecutionMode execution_mode, size_t payloads_count, uint32_t payloads[]) {
    op(SpvOpExecutionMode, 3 + payloads_count);
    ref_id(entry_point);
    literal_int(execution_mode);
    for (size_t i = 0; i < payloads_count; i++)
        literal_int(payloads[i]);
}
#undef target_data

#define target_data file_builder->debug_string_source
SpvId spvb_debug_string(SpvbFileBuilder* file_builder, const char* string) {
    op(SpvOpString, 2 + div_roundup(strlen(string) + 1, 4));
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_name(string);
    return id;
}
#undef target_data

#define target_data file_builder->debug_names
void spvb_name(SpvbFileBuilder* file_builder, SpvId id, const char* str) {
    assert(id < file_builder->bound);
    op(SpvOpName, 2 + div_roundup(strlen(str) + 1, 4));
    ref_id(id);
    literal_name(str);
}
#undef target_data

#define target_data file_builder->annotations
void spvb_decorate(SpvbFileBuilder* file_builder, SpvId target, SpvDecoration decoration, size_t extras_count, uint32_t extras[]) {
    op(SpvOpDecorate, 3 + extras_count);
    ref_id(target);
    literal_int(decoration);
    for (size_t i = 0; i < extras_count; i++)
        literal_int(extras[i]);
}

void spvb_decorate_member(SpvbFileBuilder* file_builder, SpvId target, uint32_t member, SpvDecoration decoration, size_t extras_count, uint32_t extras[]) {
    op(SpvOpMemberDecorate, 4 + extras_count);
    ref_id(target);
    literal_int(member);
    literal_int(decoration);
    for (size_t i = 0; i < extras_count; i++)
        literal_int(extras[i]);
}
#undef target_data

#define target_data file_builder->types_constants
SpvId spvb_bool_type(SpvbFileBuilder* file_builder) {
    op(SpvOpTypeBool, 2);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    return id;
}

SpvId spvb_int_type(SpvbFileBuilder* file_builder, int width, bool signed_) {
    op(SpvOpTypeInt, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_int(width);
    literal_int(signed_ ? 1 : 0);
    return id;
}

SpvId spvb_float_type(SpvbFileBuilder* file_builder, int width) {
    op(SpvOpTypeFloat, 3);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_int(width);
    return id;
}

SpvId spvb_void_type(SpvbFileBuilder* file_builder) {
    op(SpvOpTypeVoid, 2);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    return id;
}

SpvId spvb_ptr_type(SpvbFileBuilder* file_builder, SpvStorageClass storage_class, SpvId element_type) {
    op(SpvOpTypePointer, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_int(storage_class);
    ref_id(element_type);
    return id;
}

SpvId spvb_array_type(SpvbFileBuilder* file_builder, SpvId element_type, SpvId dim) {
    op(SpvOpTypeArray, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(element_type);
    ref_id(dim);
    return id;
}

SpvId spvb_runtime_array_type(SpvbFileBuilder* file_builder, SpvId element_type) {
    op(SpvOpTypeRuntimeArray, 3);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(element_type);
    return id;
}

SpvId spvb_fn_type(SpvbFileBuilder* file_builder, size_t args_count, SpvId args_types[], SpvId codom) {
    op(SpvOpTypeFunction, 3 + args_count);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(codom);
    for (size_t i = 0; i < args_count; i++)
        ref_id(args_types[i]);
    return id;
}

SpvId spvb_struct_type(SpvbFileBuilder* file_builder, SpvId id, size_t members_count, SpvId members[]) {
    op(SpvOpTypeStruct, 2 + members_count);
    ref_id(id);
    for (size_t i = 0; i < members_count; i++)
        ref_id(members[i]);
    return id;
}

SpvId spvb_vector_type(SpvbFileBuilder* file_builder, SpvId component_type, uint32_t dim) {
    op(SpvOpTypeVector, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(component_type);
    literal_int(dim);
    return id;
}

SpvId spvb_image_type(SpvbFileBuilder* file_builder, SpvId component_type, uint32_t dim, uint32_t depth, uint32_t onion, uint32_t multisample, uint32_t sampled, SpvImageFormat image_format) {
    op(SpvOpTypeImage, 9);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(component_type);
    literal_int(dim);
    literal_int(depth);
    literal_int(onion);
    literal_int(multisample);
    literal_int(sampled);
    literal_int(image_format);
    return id;
}

SpvId spvb_sampler_type(SpvbFileBuilder* file_builder) {
    op(SpvOpTypeSampler, 2);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    return id;
}

SpvId spvb_sampled_image_type(SpvbFileBuilder* file_builder, SpvId image_type) {
    op(SpvOpTypeSampledImage, 3);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(image_type);
    return id;
}

void spvb_bool_constant(SpvbFileBuilder* file_builder, SpvId result, SpvId type, bool value) {
    op(value ? SpvOpConstantTrue : SpvOpConstantFalse, 3);
    ref_id(type);
    ref_id(result);
}

void spvb_constant(SpvbFileBuilder* file_builder, SpvId result, SpvId type, size_t bit_pattern_size, uint32_t bit_pattern[]) {
    op(SpvOpConstant, 3 + bit_pattern_size);
    ref_id(type);
    ref_id(result);
    for (size_t i = 0; i < bit_pattern_size; i++)
        literal_int(bit_pattern[i]);
}

SpvId spvb_constant_composite(SpvbFileBuilder* file_builder, SpvId type, size_t ops_count, SpvId ops[]) {
    op(SpvOpConstantComposite, 3 + ops_count);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(type);
    ref_id(id);
    for (size_t i = 0; i < ops_count; i++)
        ref_id(ops[i]);
    return id;
}

SpvId spvb_constant_null(SpvbFileBuilder* file_builder, SpvId type) {
    op(SpvOpConstantNull, 3);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(type);
    ref_id(id);
    return id;
}

SpvId spvb_global_variable(SpvbFileBuilder* file_builder, SpvId id, SpvId type, SpvStorageClass storage_class, bool has_initializer, SpvId initializer) {
    op(SpvOpVariable, has_initializer ? 5 : 4);
    ref_id(type);
    ref_id(id);
    literal_int(storage_class);
    if (has_initializer)
        ref_id(initializer);
    return id;
}

SpvId spvb_undef(SpvbFileBuilder* file_builder, SpvId type) {
    op(SpvOpUndef, 3);
    ref_id(type);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    return id;
}
#undef target_data

struct SpvbFnBuilder_ {
    SpvbFileBuilder* file_builder;
    SpvId function_id;

    SpvId fn_type;
    SpvId fn_ret_type;
    struct List* bbs;

    // Contains OpFunctionParams
    SpvbSectionBuilder header;
    SpvbSectionBuilder variables;
};

SpvbFnBuilder* spvb_begin_fn(SpvbFileBuilder* file_builder, SpvId fn_id, SpvId fn_type, SpvId fn_ret_type) {
    SpvbFnBuilder* fnb = (SpvbFnBuilder*) malloc(sizeof(SpvbFnBuilder));
    *fnb = (SpvbFnBuilder) {
        .function_id = fn_id,
        .fn_type = fn_type,
        .fn_ret_type = fn_ret_type,
        .file_builder = file_builder,
        .bbs = new_list(SpvbBasicBlockBuilder*),
        .variables = new_growy(),
        .header = new_growy(),
    };
    return fnb;
}

SpvId fn_ret_type_id(SpvbFnBuilder* fnb){
    return fnb->fn_ret_type;
}

#define target_data fn_builder->header
SpvId spvb_parameter(SpvbFnBuilder* fn_builder, SpvId param_type) {
    op(SpvOpFunctionParameter, 3);
    SpvId id = spvb_fresh_id(fn_builder->file_builder);
    ref_id(param_type);
    ref_id(id);
    return id;
}
#undef target_data

#define target_data fn_builder->variables
SpvId spvb_local_variable(SpvbFnBuilder* fn_builder, SpvId type, SpvStorageClass storage_class) {
    op(SpvOpVariable, 4);
    ref_id(type);
    SpvId id = spvb_fresh_id(fn_builder->file_builder);
    ref_id(id);
    literal_int(storage_class);
    return id;
}
#undef target_data

#define target_data file_builder->fn_decls
void spvb_declare_function(SpvbFileBuilder* file_builder, SpvbFnBuilder* fn_builder) {
    op(SpvOpFunction, 5);
    ref_id(fn_builder->fn_ret_type);
    ref_id(fn_builder->function_id);
    literal_int(SpvFunctionControlMaskNone);
    ref_id(fn_builder->fn_type);

    // Includes stuff like OpFunctionParameters
    copy_section(fn_builder->header);

    assert(entries_count_list(fn_builder->bbs) == 0 && "declared functions must be empty");

    op(SpvOpFunctionEnd, 1);

    destroy_list(fn_builder->bbs);
    destroy_growy(fn_builder->header);
    destroy_growy(fn_builder->variables);
    free(fn_builder);
}
#undef target_data

struct SpvbBasicBlockBuilder_ {
    SpvbFnBuilder* fn_builder;
    SpvbSectionBuilder section_data;

    struct List* phis;
    SpvId label;
};

typedef struct {
    SpvId basic_block;
    SpvId value;
} SpvbPhiSrc;

struct SpvbPhi_ {
    SpvId type;
    SpvId value;
    struct List* preds;
};

#define target_data file_builder->fn_defs
void spvb_define_function(SpvbFileBuilder* file_builder, SpvbFnBuilder* fn_builder) {
    op(SpvOpFunction, 5);
    ref_id(fn_builder->fn_ret_type);
    ref_id(fn_builder->function_id);
    literal_int(SpvFunctionControlMaskNone);
    ref_id(fn_builder->fn_type);

    // Includes stuff like OpFunctionParameters
    copy_section(fn_builder->header);

    bool first = true;
    for (size_t i = 0; i < fn_builder->bbs->elements_count; i++) {
        op(SpvOpLabel, 2);
        SpvbBasicBlockBuilder* bb = read_list(SpvbBasicBlockBuilder*, fn_builder->bbs)[i];
        ref_id(bb->label);

        if (first) {
            // Variables are to be defined in the first BB
            copy_section(fn_builder->variables);
            first = false;
        }

        for (size_t j = 0; j < bb->phis->elements_count; j++) {
            SpvbPhi* phi = read_list(SpvbPhi*, bb->phis)[j];

            op(SpvOpPhi, 3 + 2 * phi->preds->elements_count);
            ref_id(phi->type);
            ref_id(phi->value);
            assert(phi->preds->elements_count != 0);
            for (size_t k = 0; k < phi->preds->elements_count; k++) {
                SpvbPhiSrc* pred = &read_list(SpvbPhiSrc, phi->preds)[k];
                ref_id(pred->value);
                ref_id(pred->basic_block);
            }

            destroy_list(phi->preds);
            free(phi);
        }

        copy_section(bb->section_data);

        destroy_list(bb->phis);
        destroy_growy(bb->section_data);
        free(bb);
    }

    op(SpvOpFunctionEnd, 1);

    destroy_list(fn_builder->bbs);
    destroy_growy(fn_builder->header);
    destroy_growy(fn_builder->variables);
    free(fn_builder);
}
#undef target_data

SpvbBasicBlockBuilder* spvb_begin_bb(SpvbFnBuilder* fn_builder, SpvId label) {
    SpvbBasicBlockBuilder* bbb = (SpvbBasicBlockBuilder*) malloc(sizeof(SpvbBasicBlockBuilder));
    *bbb = (SpvbBasicBlockBuilder) {
            .fn_builder = fn_builder,
            .label = label,
            .phis = new_list(SpvbPhi*),
            .section_data = new_growy()
    };
    return bbb;
}

void spvb_add_bb(SpvbFnBuilder* fn_builder, SpvbBasicBlockBuilder* bb_builder) {
    append_list(SpvbBasicBlockBuilder*, fn_builder->bbs, bb_builder);
}

SpvId get_block_builder_id(SpvbBasicBlockBuilder* basic_block_builder) {
    return basic_block_builder->label;
}
SpvbPhi* spvb_add_phi(SpvbBasicBlockBuilder* bb_builder, SpvId type, SpvId id) {
    SpvbPhi* phi = malloc(sizeof(SpvbPhi));
    phi->preds = new_list(SpvbPhiSrc);
    phi->value = id;
    phi->type = type;
    append_list(SpvbPhi*, bb_builder->phis, phi);
    return phi;
}

void spvb_add_phi_source(SpvbPhi* phi, SpvId source_block, SpvId value) {
    SpvbPhiSrc op = { .value = value, .basic_block = source_block };
    append_list(SpvbPhi, phi->preds, op);
}

struct List* spbv_get_phis(SpvbBasicBlockBuilder* bb_builder) {
    return bb_builder->phis;
}

// It is tiresome to pass the context over and over again. Let's not !
// We use this macro to save us some typing
#define target_data bb_builder->section_data
SpvId spvb_composite(SpvbBasicBlockBuilder* bb_builder, SpvId aggregate_t, size_t elements_count, SpvId elements[]) {
    op(SpvOpCompositeConstruct, 3u + elements_count);
    ref_id(aggregate_t);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    for (size_t i = 0; i < elements_count; i++)
        ref_id(elements[i]);
    return id;
}

SpvId spvb_select(SpvbBasicBlockBuilder* bb_builder, SpvId type, SpvId condition, SpvId if_true, SpvId if_false) {
    op(SpvOpSelect, 6);
    ref_id(type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(condition);
    ref_id(if_true);
    ref_id(if_false);
    return id;
}

SpvId spvb_extract(SpvbBasicBlockBuilder* bb_builder, SpvId target_type, SpvId composite, size_t indices_count, uint32_t indices[]) {
    op(SpvOpCompositeExtract, 4u + indices_count);
    ref_id(target_type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(composite);
    for (size_t i = 0; i < indices_count; i++)
        literal_int(indices[i]);
    return id;
}

SpvId spvb_insert(SpvbBasicBlockBuilder* bb_builder, SpvId target_type, SpvId object, SpvId composite, size_t indices_count, uint32_t indices[]) {
    op(SpvOpCompositeInsert, 5 + indices_count);
    ref_id(target_type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(object);
    ref_id(composite);
    for (size_t i = 0; i < indices_count; i++)
        literal_int(indices[i]);
    return id;
}

SpvId spvb_vector_extract_dynamic(SpvbBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId index) {
    op(SpvOpVectorExtractDynamic, 5);
    ref_id(target_type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(vector);
    ref_id(index);
    return id;
}

SpvId spvb_vector_insert_dynamic(SpvbBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId component, SpvId index) {
    op(SpvOpVectorInsertDynamic, 6);
    ref_id(target_type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(vector);
    ref_id(component);
    ref_id(index);
    return id;
}

SpvId spvb_access_chain(SpvbBasicBlockBuilder* bb_builder, SpvId target_type, SpvId element, size_t indices_count, SpvId indices[]) {
    op(SpvOpAccessChain, 4 + indices_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(target_type);
    ref_id(id);
    ref_id(element);
    for (size_t i = 0; i < indices_count; i++)
        ref_id(indices[i]);
    return id;
}

SpvId spvb_ptr_access_chain(SpvbBasicBlockBuilder* bb_builder, SpvId target_type, SpvId base, SpvId element, size_t indices_count, SpvId indices[]) {
    op(SpvOpPtrAccessChain, 5 + indices_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(target_type);
    ref_id(id);
    ref_id(base);
    ref_id(element);
    for (size_t i = 0; i < indices_count; i++)
        ref_id(indices[i]);
    return id;
}

SpvId spvb_load(SpvbBasicBlockBuilder* bb_builder, SpvId target_type, SpvId pointer, size_t operands_count, uint32_t operands[]) {
    op(SpvOpLoad, 4 + operands_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(target_type);
    ref_id(id);
    ref_id(pointer);
    for (size_t i = 0; i < operands_count; i++)
        literal_int(operands[i]);
    return id;
}

void spvb_store(SpvbBasicBlockBuilder* bb_builder, SpvId value, SpvId pointer, size_t operands_count, uint32_t operands[]) {
    op(SpvOpStore, 3 + operands_count);
    ref_id(pointer);
    ref_id(value);
    for (size_t i = 0; i < operands_count; i++)
        literal_int(operands[i]);
}

SpvId spvb_op(SpvbBasicBlockBuilder* bb_builder, SpvOp op, SpvId result_type, size_t operands_count, SpvId operands[]) {
    op(op, 3 + operands_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    for (size_t i = 0; i < operands_count; i++)
        ref_id(operands[i]);
    return id;
}

SpvId spvb_elect(SpvbBasicBlockBuilder* bb_builder, SpvId result_type, SpvId scope) {
    op(SpvOpGroupNonUniformElect, 4);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(scope);
    return id;
}

SpvId spvb_ballot(SpvbBasicBlockBuilder* bb_builder, SpvId result_type, SpvId predicate, SpvId scope) {
    op(SpvOpGroupNonUniformBallot, 5);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(scope);
    ref_id(predicate);
    return id;
}

SpvId spvb_broadcast_first(SpvbBasicBlockBuilder* bb_builder, SpvId result_type, SpvId value, SpvId scope) {
    op(SpvOpGroupNonUniformBroadcastFirst, 5);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(scope);
    ref_id(value);
    return id;
}

SpvId spvb_shuffle(SpvbBasicBlockBuilder* bb_builder, SpvId result_type, SpvId scope, SpvId value, SpvId id) {
    op(SpvOpGroupNonUniformShuffle, 6);
    SpvId rid = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(rid);
    ref_id(scope);
    ref_id(value);
    ref_id(id);
    return rid;
}

SpvId spvb_non_uniform_iadd(SpvbBasicBlockBuilder* bb_builder, SpvId result_type, SpvId value, SpvId scope, SpvGroupOperation group_op, SpvId* cluster_size) {
    op(SpvOpGroupNonUniformIAdd, cluster_size ? 7 : 6);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(scope);
    literal_int(group_op);
    ref_id(value);
    if (cluster_size)
        ref_id(*cluster_size);
    return id;
}

void spvb_branch(SpvbBasicBlockBuilder* bb_builder, SpvId target) {
    op(SpvOpBranch, 2);
    ref_id(target);
}

void spvb_branch_conditional(SpvbBasicBlockBuilder* bb_builder, SpvId condition, SpvId true_target, SpvId false_target) {
    op(SpvOpBranchConditional, 4);
    ref_id(condition);
    ref_id(true_target);
    ref_id(false_target);
}

void spvb_switch(SpvbBasicBlockBuilder* bb_builder, SpvId selector, SpvId default_target, size_t targets_and_literals_size, SpvId* targets_and_literals) {
    op(SpvOpSwitch, 3 + targets_and_literals_size);
    ref_id(selector);
    ref_id(default_target);
    for (size_t i = 0; i < targets_and_literals_size; i++) {
        literal_int(targets_and_literals[i]);
    }
}

void spvb_selection_merge(SpvbBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvSelectionControlMask selection_control) {
    op(SpvOpSelectionMerge, 3);
    ref_id(merge_bb);
    literal_int(selection_control);
}

void spvb_loop_merge(SpvbBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvId continue_bb, SpvLoopControlMask loop_control, size_t loop_control_ops_count, uint32_t loop_control_ops[]) {
    op(SpvOpLoopMerge, 4 + loop_control_ops_count);
    ref_id(merge_bb);
    ref_id(continue_bb);
    literal_int(loop_control);

    for (size_t i = 0; i < loop_control_ops_count; i++)
        literal_int(loop_control_ops[i]);
}

SpvId spvb_call(SpvbBasicBlockBuilder* bb_builder, SpvId return_type, SpvId callee, size_t arguments_count, SpvId arguments[]) {
    op(SpvOpFunctionCall, 4u + arguments_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(return_type);
    ref_id(id);
    ref_id(callee);
    for (size_t i = 0; i < arguments_count; i++)
        ref_id(arguments[i]);
    return id;
}

SpvId spvb_ext_instruction(SpvbBasicBlockBuilder* bb_builder, SpvId return_type, SpvId set, uint32_t instruction, size_t arguments_count, SpvId arguments[]) {
    op(SpvOpExtInst, 5 + arguments_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(return_type);
    ref_id(id);
    ref_id(set);
    literal_int(instruction);
    for (size_t i = 0; i < arguments_count; i++)
        ref_id(arguments[i]);
    return id;
}

void spvb_return_void(SpvbBasicBlockBuilder* bb_builder) {
    op(SpvOpReturn, 1);
}

void spvb_return_value(SpvbBasicBlockBuilder* bb_builder, SpvId value) {
    op(SpvOpReturnValue, 2);
    ref_id(value);
}

void spvb_unreachable(SpvbBasicBlockBuilder* bb_builder) {
    op(SpvOpUnreachable, 1);
}
#undef target_data
