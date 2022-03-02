#ifndef SHADY_SPIRV_BUILDER_H
#define SHADY_SPIRV_BUILDER_H

#include <spirv/unified1/spirv.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct List* SpvSectionBuilder;

struct SpvBasicBlockBuilder;
struct SpvFnBuilder;
struct SpvFileBuilder;

SpvId undef(struct SpvBasicBlockBuilder* bb_builder, SpvId type);
SpvId composite(struct SpvBasicBlockBuilder* bb_builder, SpvId aggregate_t, size_t elements_count, SpvId elements[]);
SpvId extract(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId composite, size_t indices_count, uint32_t indices[]);
SpvId insert(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId object, SpvId composite, size_t indices_count, uint32_t indices[]);
SpvId vector_extract_dynamic(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId index);
SpvId vector_insert_dynamic(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId component, SpvId index);
/// Used for almost all conversion operations
SpvId convert(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId target_type, SpvId value);
SpvId access_chain(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId element, size_t indices_count, SpvId indices[]);
SpvId ptr_access_chain(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId base, SpvId element, size_t indices_count, SpvId indices[]);
SpvId load(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId pointer, size_t operands_count, uint32_t operands[]);
void store(struct SpvBasicBlockBuilder* bb_builder, SpvId value, SpvId pointer, size_t operands_count, uint32_t operands[]);
SpvId binop(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId result_type, SpvId lhs, SpvId rhs);
void branch(struct SpvBasicBlockBuilder* bb_builder, SpvId target);
void branch_conditional(struct SpvBasicBlockBuilder* bb_builder, SpvId condition, SpvId true_target, SpvId false_target);
void selection_merge(struct SpvBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvSelectionControlMask selection_control) ;
void loop_merge(struct SpvBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvId continue_bb, SpvLoopControlMask loop_control, size_t loop_control_ops_count, uint32_t loop_control_ops[]);
SpvId call(struct SpvBasicBlockBuilder* bb_builder, SpvId return_type, SpvId callee, size_t arguments_count, SpvId arguments[]);
SpvId ext_instruction(struct SpvBasicBlockBuilder* bb_builder, SpvId return_type, SpvId set, uint32_t instruction, size_t arguments_count, SpvId arguments[]);
void return_void(struct SpvBasicBlockBuilder* bb_builder) ;
void return_value(struct SpvBasicBlockBuilder* bb_builder, SpvId value);
void unreachable(struct SpvBasicBlockBuilder* bb_builder);

SpvId parameter(struct SpvFnBuilder* fn_builder, SpvId param_type);
SpvId local_variable(struct SpvFnBuilder* fn_builder, SpvId type, SpvStorageClass storage_class);

struct SpvFileBuilder;

void name(struct SpvFileBuilder* file_builder, SpvId id, const char* str);

SpvId declare_void_type(struct SpvFileBuilder* file_builder);
SpvId declare_bool_type(struct SpvFileBuilder* file_builder);
SpvId declare_int_type(struct SpvFileBuilder* file_builder, int width, bool signed_);
SpvId declare_float_type(struct SpvFileBuilder* file_builder, int width);
SpvId declare_ptr_type(struct SpvFileBuilder* file_builder, SpvStorageClass storage_class, SpvId element_type);
SpvId declare_array_type(struct SpvFileBuilder* file_builder, SpvId element_type, SpvId dim);
SpvId declare_fn_type(struct SpvFileBuilder* file_builder, size_t args_count, SpvId args_types[], SpvId codom);
SpvId declare_struct_type(struct SpvFileBuilder* file_builder, size_t members_count, SpvId members[]);
SpvId declare_vector_type(struct SpvFileBuilder* file_builder, SpvId component_type, uint32_t dim);
SpvId bool_constant(struct SpvFileBuilder* file_builder, SpvId type, bool value);
SpvId constant(struct SpvFileBuilder* file_builder, SpvId type, size_t bit_pattern_size, uint32_t bit_pattern[]);
SpvId constant_composite(struct SpvFileBuilder* file_builder, SpvId type, size_t ops_count, SpvId ops[]);
SpvId global_variable(struct SpvFileBuilder* file_builder, SpvId type, SpvStorageClass storage_class);

void decorate(struct SpvFileBuilder* file_builder, SpvId target, SpvDecoration decoration, size_t extras_count, uint32_t extras[]);
void decorate_member(struct SpvFileBuilder* file_builder, SpvId target, uint32_t member, SpvDecoration decoration, size_t extras_count, uint32_t extras[]);

SpvId debug_string(struct SpvFileBuilder* file_builder, const char* string);

SpvId define_function(struct SpvFileBuilder* file_builder, struct SpvFnBuilder* fn_builder);

void declare_entry_point(struct SpvFileBuilder* file_builder, SpvExecutionModel execution_model, SpvId entry_point, const char* name, size_t interface_elements_count, SpvId interface_elements[]);
void execution_mode(struct SpvFileBuilder* file_builder, SpvId entry_point, SpvExecutionMode execution_mode, size_t payloads_count, uint32_t payloads[]);
void capability(struct SpvFileBuilder* file_builder, SpvCapability cap);

void extension(struct SpvFileBuilder* file_builder, const char* name);

SpvId extended_import(struct SpvFileBuilder* file_builder, const char* name);

#endif
