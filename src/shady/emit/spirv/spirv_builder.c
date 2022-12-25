#include "spirv_builder.h"

#include "list.h"

#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

typedef struct List* SpvSectionBuilder;

struct PhiOp {
    SpvId basic_block;
    SpvId value;
};

struct Phi {
    SpvId type;
    SpvId value;
    struct List* preds;
};

struct SpvFnBuilder {
    struct SpvFileBuilder* file_builder;
    SpvId function_id;

    SpvId fn_type;
    SpvId fn_ret_type;
    struct List* bbs;

    // Contains OpFunctionParams
    SpvSectionBuilder header;
    SpvSectionBuilder variables;
};

struct SpvBasicBlockBuilder {
    struct SpvFnBuilder* fn_builder;
    struct List* section_data;

    struct List* phis;
    SpvId label;
};

struct SpvFileBuilder {
    SpvAddressingModel addressing_model;
    SpvMemoryModel memory_model;

    uint32_t bound;

    struct {
        uint8_t major;
        uint8_t minor;
    } version;

    // Ordered as per https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.pdf#subsection.2.4
    SpvSectionBuilder capabilities;
    SpvSectionBuilder extensions;
    SpvSectionBuilder ext_inst_import;
    SpvSectionBuilder entry_points;
    SpvSectionBuilder execution_modes;
    SpvSectionBuilder debug_string_source;
    SpvSectionBuilder debug_names;
    SpvSectionBuilder debug_module_processed;
    SpvSectionBuilder annotations;
    SpvSectionBuilder types_constants;
    SpvSectionBuilder fn_decls;
    SpvSectionBuilder fn_defs;
};

SpvId spvb_fresh_id(struct SpvFileBuilder* file_builder) {
    return file_builder->bound++;
}

inline static int div_roundup(int a, int b) {
    if (a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

inline static void output_word(SpvSectionBuilder data, uint32_t word) {
    append_list(uint32_t, data, word);
}

#define op(opcode, size) op_(target_data, opcode, size)
inline static void op_(SpvSectionBuilder data, SpvOp op, int ops_size) {
    uint32_t lower = op & 0xFFFFu;
    uint32_t upper = (ops_size << 16) & 0xFFFF0000u;
    output_word(data, lower | upper);
}

#define ref_id(i) ref_id_(target_data, i)
inline static void ref_id_(SpvSectionBuilder data, SpvId id) {
    assert(id != 0);
    output_word(data, id);
}

#define literal_name(str) literal_name_(target_data, str)
inline static void literal_name_(SpvSectionBuilder data, const char* str) {
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
inline static void literal_int_(SpvSectionBuilder data, uint32_t i) {
    output_word(data, i);
}

#define copy_section(section) copy_section_(target_data, section)
inline static void copy_section_(SpvSectionBuilder target, SpvSectionBuilder source) {
    for (size_t i = 0; i < source->elements_count; i++)
        literal_int_(target, read_list(uint32_t, source)[i]);
}

// It is tiresome to pass the context over and over again. Let's not !
// We use this macro to save us some typing
#define target_data bb_builder->section_data

SpvId spvb_undef(struct SpvBasicBlockBuilder* bb_builder, SpvId type) {
    op(SpvOpUndef, 3);
    ref_id(type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    return id;
}

SpvId spvb_composite(struct SpvBasicBlockBuilder* bb_builder, SpvId aggregate_t, size_t elements_count, SpvId elements[]) {
    op(SpvOpCompositeConstruct, 3u + elements_count);
    ref_id(aggregate_t);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    for (size_t i = 0; i < elements_count; i++)
        ref_id(elements[i]);
    return id;
}

SpvId spvb_select(struct SpvBasicBlockBuilder* bb_builder, SpvId type, SpvId condition, SpvId if_true, SpvId if_false) {
    op(SpvOpSelect, 6);
    ref_id(type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(condition);
    ref_id(if_true);
    ref_id(if_false);
    return id;
}

SpvId spvb_extract(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId composite, size_t indices_count, uint32_t indices[]) {
    op(SpvOpCompositeExtract, 4u + indices_count);
    ref_id(target_type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(composite);
    for (size_t i = 0; i < indices_count; i++)
        literal_int(indices[i]);
    return id;
}

SpvId spvb_insert(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId object, SpvId composite, size_t indices_count, uint32_t indices[]) {
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

SpvId spvb_vector_extract_dynamic(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId index) {
    op(SpvOpVectorExtractDynamic, 5);
    ref_id(target_type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(vector);
    ref_id(index);
    return id;
}

SpvId spvb_vector_insert_dynamic(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId component, SpvId index) {
    op(SpvOpVectorInsertDynamic, 6);
    ref_id(target_type);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(id);
    ref_id(vector);
    ref_id(component);
    ref_id(index);
    return id;
}

// Used for almost all conversion operations
SpvId spvb_convert(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId target_type, SpvId value) {
    op(op, 4);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(target_type);
    ref_id(id);
    ref_id(value);
    return id;
}

SpvId spvb_access_chain(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId element, size_t indices_count, SpvId indices[]) {
    op(SpvOpAccessChain, 4 + indices_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(target_type);
    ref_id(id);
    ref_id(element);
    for (size_t i = 0; i < indices_count; i++)
        ref_id(indices[i]);
    return id;
}

SpvId spvb_ptr_access_chain(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId base, SpvId element, size_t indices_count, SpvId indices[]) {
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

SpvId spvb_load(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId pointer, size_t operands_count, uint32_t operands[]) {
    op(SpvOpLoad, 4 + operands_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(target_type);
    ref_id(id);
    ref_id(pointer);
    for (size_t i = 0; i < operands_count; i++)
        literal_int(operands[i]);
    return id;
}

void spvb_store(struct SpvBasicBlockBuilder* bb_builder, SpvId value, SpvId pointer, size_t operands_count, uint32_t operands[]) {
    op(SpvOpStore, 3 + operands_count);
    ref_id(pointer);
    ref_id(value);
    for (size_t i = 0; i < operands_count; i++)
        literal_int(operands[i]);
}

SpvId spvb_binop(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId result_type, SpvId lhs, SpvId rhs) {
    op(op, 5);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(lhs);
    ref_id(rhs);
    return id;
}

SpvId spvb_unop(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId result_type, SpvId value) {
    op(op, 4);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(value);
    return id;
}

SpvId spvb_elect(struct SpvBasicBlockBuilder* bb_builder, SpvId result_type, SpvId scope) {
    op(SpvOpGroupNonUniformElect, 4);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(scope);
    return id;
}

SpvId spvb_ballot(struct SpvBasicBlockBuilder* bb_builder, SpvId result_type, SpvId predicate, SpvId scope) {
    op(SpvOpGroupNonUniformBallot, 5);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(scope);
    ref_id(predicate);
    return id;
}

SpvId spvb_broadcast_first(struct SpvBasicBlockBuilder* bb_builder, SpvId result_type, SpvId value, SpvId scope) {
    op(SpvOpGroupNonUniformBroadcastFirst, 5);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(result_type);
    ref_id(id);
    ref_id(scope);
    ref_id(value);
    return id;
}

SpvId spvb_non_uniform_iadd(struct SpvBasicBlockBuilder* bb_builder, SpvId result_type, SpvId value, SpvId scope, SpvGroupOperation group_op, SpvId* cluster_size) {
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

void spvb_branch(struct SpvBasicBlockBuilder* bb_builder, SpvId target) {
    op(SpvOpBranch, 2);
    ref_id(target);
}

void spvb_branch_conditional(struct SpvBasicBlockBuilder* bb_builder, SpvId condition, SpvId true_target, SpvId false_target) {
    op(SpvOpBranchConditional, 4);
    ref_id(condition);
    ref_id(true_target);
    ref_id(false_target);
}

void spvb_switch(struct SpvBasicBlockBuilder* bb_builder, SpvId selector, SpvId default_target, size_t targets_count, SpvId* targets) {
    op(SpvOpSwitch, 3 + targets_count * 2);
    ref_id(selector);
    ref_id(default_target);
    for (size_t i = 0; i < targets_count; i++) {
        literal_int(targets[i * 2]);
        ref_id(targets[i * 2 + 1]);
    }
}

void spvb_selection_merge(struct SpvBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvSelectionControlMask selection_control) {
    op(SpvOpSelectionMerge, 3);
    ref_id(merge_bb);
    literal_int(selection_control);
}

void spvb_loop_merge(struct SpvBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvId continue_bb, SpvLoopControlMask loop_control, size_t loop_control_ops_count, uint32_t loop_control_ops[]) {
    op(SpvOpLoopMerge, 4 + loop_control_ops_count);
    ref_id(merge_bb);
    ref_id(continue_bb);
    literal_int(loop_control);

    for (size_t i = 0; i < loop_control_ops_count; i++)
        literal_int(loop_control_ops[i]);
}

SpvId spvb_call(struct SpvBasicBlockBuilder* bb_builder, SpvId return_type, SpvId callee, size_t arguments_count, SpvId arguments[]) {
    op(SpvOpFunctionCall, 4u + arguments_count);
    SpvId id = spvb_fresh_id(bb_builder->fn_builder->file_builder);
    ref_id(return_type);
    ref_id(id);
    ref_id(callee);
    for (size_t i = 0; i < arguments_count; i++)
        ref_id(arguments[i]);
    return id;
}

SpvId spvb_ext_instruction(struct SpvBasicBlockBuilder* bb_builder, SpvId return_type, SpvId set, uint32_t instruction, size_t arguments_count, SpvId arguments[]) {
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

void spvb_return_void(struct SpvBasicBlockBuilder* bb_builder) {
    op(SpvOpReturn, 1);
}

void spvb_return_value(struct SpvBasicBlockBuilder* bb_builder, SpvId value) {
    op(SpvOpReturnValue, 2);
    ref_id(value);
}

void spvb_unreachable(struct SpvBasicBlockBuilder* bb_builder) {
    op(SpvOpUnreachable, 1);
}

#undef target_data
#define target_data fn_builder->header

SpvId spvb_parameter(struct SpvFnBuilder* fn_builder, SpvId param_type) {
    op(SpvOpFunctionParameter, 3);
    SpvId id = spvb_fresh_id(fn_builder->file_builder);
    ref_id(param_type);
    ref_id(id);
    return id;
}

#undef target_data
#define target_data fn_builder->variables

SpvId spvb_local_variable(struct SpvFnBuilder* fn_builder, SpvId type, SpvStorageClass storage_class) {
    op(SpvOpVariable, 4);
    ref_id(type);
    SpvId id = spvb_fresh_id(fn_builder->file_builder);
    ref_id(id);
    literal_int(storage_class);
    return id;
}

#undef target_data
#define target_data file_builder->debug_names

void spvb_name(struct SpvFileBuilder* file_builder, SpvId id, const char* str) {
    assert(id < file_builder->bound);
    op(SpvOpName, 2 + div_roundup(strlen(str) + 1, 4));
    ref_id(id);
    literal_name(str);
}

#undef target_data
#define target_data file_builder->types_constants

SpvId spvb_bool_type(struct SpvFileBuilder* file_builder) {
    op(SpvOpTypeBool, 2);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    return id;
}

SpvId spvb_int_type(struct SpvFileBuilder* file_builder, int width, bool signed_) {
    op(SpvOpTypeInt, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_int(width);
    literal_int(signed_ ? 1 : 0);
    return id;
}

SpvId spvb_float_type(struct SpvFileBuilder* file_builder, int width) {
    op(SpvOpTypeFloat, 3);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_int(width);
    return id;
}

SpvId spvb_void_type(struct SpvFileBuilder* file_builder) {
    op(SpvOpTypeVoid, 2);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    return id;
}

SpvId spvb_ptr_type(struct SpvFileBuilder* file_builder, SpvStorageClass storage_class, SpvId element_type) {
    op(SpvOpTypePointer, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_int(storage_class);
    ref_id(element_type);
    return id;
}

SpvId spvb_array_type(struct SpvFileBuilder* file_builder, SpvId element_type, SpvId dim) {
    op(SpvOpTypeArray, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(element_type);
    ref_id(dim);
    return id;
}

SpvId spvb_runtime_array_type(struct SpvFileBuilder* file_builder, SpvId element_type) {
    op(SpvOpTypeRuntimeArray, 3);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(element_type);
    return id;
}

SpvId spvb_fn_type(struct SpvFileBuilder* file_builder, size_t args_count, SpvId args_types[], SpvId codom) {
    op(SpvOpTypeFunction, 3 + args_count);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(codom);
    for (size_t i = 0; i < args_count; i++)
        ref_id(args_types[i]);
    return id;
}

SpvId spvb_struct_type(struct SpvFileBuilder* file_builder, SpvId id, size_t members_count, SpvId members[]) {
    op(SpvOpTypeStruct, 2 + members_count);
    ref_id(id);
    for (size_t i = 0; i < members_count; i++)
        ref_id(members[i]);
    return id;
}

SpvId spvb_vector_type(struct SpvFileBuilder* file_builder, SpvId component_type, uint32_t dim) {
    op(SpvOpTypeVector, 4);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    ref_id(component_type);
    literal_int(dim);
    return id;
}

void spvb_bool_constant(struct SpvFileBuilder* file_builder, SpvId result, SpvId type, bool value) {
    op(value ? SpvOpConstantTrue : SpvOpConstantFalse, 3);
    ref_id(type);
    ref_id(result);
}

void spvb_constant(struct SpvFileBuilder* file_builder, SpvId result, SpvId type, size_t bit_pattern_size, uint32_t bit_pattern[]) {
    op(SpvOpConstant, 3 + bit_pattern_size);
    ref_id(type);
    ref_id(result);
    for (size_t i = 0; i < bit_pattern_size; i++)
        literal_int(bit_pattern[i]);
}

SpvId spvb_constant_composite(struct SpvFileBuilder* file_builder, SpvId type, size_t ops_count, SpvId ops[]) {
    op(SpvOpConstantComposite, 3 + ops_count);
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(type);
    ref_id(id);
    for (size_t i = 0; i < ops_count; i++)
        ref_id(ops[i]);
    return id;
}

SpvId spvb_global_variable(struct SpvFileBuilder* file_builder, SpvId id, SpvId type, SpvStorageClass storage_class, bool has_initializer, SpvId initializer) {
    op(SpvOpVariable, has_initializer ? 5 : 4);
    ref_id(type);
    ref_id(id);
    literal_int(storage_class);
    if (has_initializer)
        ref_id(initializer);
    return id;
}

#undef target_data
#define target_data file_builder->annotations

void spvb_decorate(struct SpvFileBuilder* file_builder, SpvId target, SpvDecoration decoration, size_t extras_count, uint32_t extras[]) {
    op(SpvOpDecorate, 3 + extras_count);
    ref_id(target);
    literal_int(decoration);
    for (size_t i = 0; i < extras_count; i++)
        literal_int(extras[i]);
}

void spvb_decorate_member(struct SpvFileBuilder* file_builder, SpvId target, uint32_t member, SpvDecoration decoration, size_t extras_count, uint32_t extras[]) {
    op(SpvOpMemberDecorate, 4 + extras_count);
    ref_id(target);
    literal_int(member);
    literal_int(decoration);
    for (size_t i = 0; i < extras_count; i++)
        literal_int(extras[i]);
}

#undef target_data
#define target_data file_builder->debug_string_source

SpvId spvb_debug_string(struct SpvFileBuilder* file_builder, const char* string) {
    op(SpvOpString, 2 + div_roundup(strlen(string) + 1, 4));
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_name(string);
    return id;
}

#undef target_data
#define target_data file_builder->fn_defs

void spvb_define_function(struct SpvFileBuilder* file_builder, struct SpvFnBuilder* fn_builder) {
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
        struct SpvBasicBlockBuilder* bb = read_list(struct SpvBasicBlockBuilder*, fn_builder->bbs)[i];
        ref_id(bb->label);

        if (first) {
            // Variables are to be defined in the first BB
            copy_section(fn_builder->variables);
            first = false;
        }

        for (size_t j = 0; j < bb->phis->elements_count; j++) {
            struct Phi* phi = read_list(struct Phi*, bb->phis)[j];

            op(SpvOpPhi, 3 + 2 * phi->preds->elements_count);
            ref_id(phi->type);
            ref_id(phi->value);
            assert(phi->preds->elements_count != 0);
            for (size_t k = 0; k < phi->preds->elements_count; k++) {
                struct PhiOp* pred = &read_list(struct PhiOp, phi->preds)[k];
                ref_id(pred->value);
                ref_id(pred->basic_block);
            }

            destroy_list(phi->preds);
            free(phi);
        }

        copy_section(bb->section_data);

        destroy_list(bb->phis);
        destroy_list(bb->section_data);
        free(bb);
    }

    op(SpvOpFunctionEnd, 1);

    destroy_list(fn_builder->bbs);
    destroy_list(fn_builder->header);
    destroy_list(fn_builder->variables);
    free(fn_builder);
}

#undef target_data
#define target_data file_builder->entry_points

void spvb_entry_point(struct SpvFileBuilder* file_builder, SpvExecutionModel execution_model, SpvId entry_point, const char* name, size_t interface_elements_count, SpvId interface_elements[]) {
    op(SpvOpEntryPoint, 3 + div_roundup(strlen(name) + 1, 4) + interface_elements_count);
    literal_int(execution_model);
    ref_id(entry_point);
    literal_name(name);
    for (size_t i = 0; i < interface_elements_count; i++)
        ref_id(interface_elements[i]);
}

#undef target_data
#define target_data file_builder->execution_modes

void spvb_execution_mode(struct SpvFileBuilder* file_builder, SpvId entry_point, SpvExecutionMode execution_mode, size_t payloads_count, uint32_t payloads[]) {
    op(SpvOpExecutionMode, 3 + payloads_count);
    ref_id(entry_point);
    literal_int(execution_mode);
    for (size_t i = 0; i < payloads_count; i++)
        literal_int(payloads[i]);
}

#undef target_data
#define target_data file_builder->capabilities

void spvb_capability(struct SpvFileBuilder* file_builder, SpvCapability cap) {
    op(SpvOpCapability, 2);
    literal_int(cap);
}

#undef target_data
#define target_data file_builder->extensions

void spvb_extension(struct SpvFileBuilder* file_builder, const char* name) {
    op(SpvOpExtension, 1 + div_roundup(strlen(name) + 1, 4));
    literal_name(name);
}

#undef target_data
#define target_data file_builder->ext_inst_import

SpvId spvb_extended_import(struct SpvFileBuilder* file_builder, const char* name) {
    op(SpvOpExtInstImport, 2 + div_roundup(strlen(name) + 1, 4));
    SpvId id = spvb_fresh_id(file_builder);
    ref_id(id);
    literal_name(name);
    return id;
}

/*void output_word_le(struct List* output, uint32_t word) {
    char c;
    c = (char)((word >> 0) & 0xFFu);
    append_list(char, output, c);
    c = (char)((word >> 8) & 0xFFu);
    append_list(char, output, c);
    c = (char)((word >> 16) & 0xFFu);
    append_list(char, output, c);
    c = (char)((word >> 24) & 0xFFu);
    append_list(char, output, c);
}*/

#undef target_data
#define target_data final_output

struct Phi* spvb_add_phi(struct SpvBasicBlockBuilder* bb_builder, SpvId type, SpvId id) {
    struct Phi* phi = malloc(sizeof(struct Phi));
    phi->preds = new_list(struct PhiOp);
    phi->value = id;
    phi->type = type;
    append_list(struct Phi*, bb_builder->phis, phi);
    return phi;
}

void spvb_add_phi_source(struct Phi* phi, SpvId source_block, SpvId value) {
    struct PhiOp op = { .value = value, .basic_block = source_block };
    append_list(struct Phi, phi->preds, op);
}

struct List* spbv_get_phis(struct SpvBasicBlockBuilder* bb_builder) {
    return bb_builder->phis;
}

SpvId get_block_builder_id(struct SpvBasicBlockBuilder* basic_block_builder) {
    return basic_block_builder->label;
}

#define SHADY_GENERATOR_MAGIC_NUMBER 35

inline static void merge_sections(SpvSectionBuilder final_output, struct SpvFileBuilder* file_builder) {
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

void spvb_build_function(struct SpvFileBuilder* file_builder);

struct SpvFileBuilder* spvb_begin() {
    struct SpvFileBuilder* file_builder = (struct SpvFileBuilder*) malloc(sizeof(struct SpvFileBuilder));
    *file_builder = (struct SpvFileBuilder) {
        .bound = 1,
        .capabilities = new_list(uint32_t),
        .extensions = new_list(uint32_t),
        .ext_inst_import = new_list(uint32_t),
        .entry_points = new_list(uint32_t),
        .execution_modes = new_list(uint32_t),
        .debug_string_source = new_list(uint32_t),
        .debug_names = new_list(uint32_t),
        .debug_module_processed = new_list(uint32_t),
        .annotations = new_list(uint32_t),
        .types_constants = new_list(uint32_t),
        .fn_decls = new_list(uint32_t),
        .fn_defs = new_list(uint32_t),
    };
    return file_builder;
}

void spvb_set_version(struct SpvFileBuilder* file_builder, uint8_t major, uint8_t minor) {
    file_builder->version.major = major;
    file_builder->version.minor = minor;
}

void spvb_finish(struct SpvFileBuilder* file_builder, SpvSectionBuilder output) {
    merge_sections(output, file_builder);

    destroy_list(file_builder->fn_defs);
    destroy_list(file_builder->fn_decls);
    destroy_list(file_builder->types_constants);
    destroy_list(file_builder->annotations);
    destroy_list(file_builder->debug_module_processed);
    destroy_list(file_builder->debug_names);
    destroy_list(file_builder->debug_string_source);
    destroy_list(file_builder->execution_modes);
    destroy_list(file_builder->entry_points);
    destroy_list(file_builder->ext_inst_import);
    destroy_list(file_builder->extensions);
    destroy_list(file_builder->capabilities);

    free(file_builder);
}

struct SpvFnBuilder* spvb_begin_fn(struct SpvFileBuilder* file_builder, SpvId fn_id, SpvId fn_type, SpvId fn_ret_type) {
    struct SpvFnBuilder* fnb = (struct SpvFnBuilder*) malloc(sizeof(struct SpvFnBuilder));
    *fnb = (struct SpvFnBuilder) {
        .function_id = fn_id,
        .fn_type = fn_type,
        .fn_ret_type = fn_ret_type,
        .file_builder = file_builder,
        .bbs = new_list(struct SpvBasicBlockBuilder*),
        .variables = new_list(uint32_t),
        .header = new_list(uint32_t),
    };
    return fnb;
}

struct SpvBasicBlockBuilder* spvb_begin_bb(struct SpvFnBuilder* fn_builder, SpvId label) {
    struct SpvBasicBlockBuilder* bbb = (struct SpvBasicBlockBuilder*) malloc(sizeof(struct SpvBasicBlockBuilder));
    *bbb = (struct SpvBasicBlockBuilder) {
        .fn_builder = fn_builder,
        .label = label,
        .phis = new_list(struct Phi*),
        .section_data = new_list(uint32_t)
    };
    return bbb;
}

void spvb_add_bb(struct SpvFnBuilder* fn_builder, struct SpvBasicBlockBuilder* bb_builder) {
    append_list(struct SpvBasicBlockBuilder*, fn_builder->bbs, bb_builder);
}

SpvId fn_ret_type_id(struct SpvFnBuilder* fnb){
    return fnb->fn_ret_type;
}
