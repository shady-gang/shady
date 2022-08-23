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

SpvId spvb_undef(struct SpvBasicBlockBuilder* bb_builder, SpvId type);
SpvId spvb_composite(struct SpvBasicBlockBuilder* bb_builder, SpvId aggregate_t, size_t elements_count, SpvId elements[]);
SpvId spvb_select(struct SpvBasicBlockBuilder* bb_builder, SpvId type, SpvId condition, SpvId if_true, SpvId if_false);
SpvId spvb_extract(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId composite, size_t indices_count, uint32_t indices[]);
SpvId spvb_insert(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId object, SpvId composite, size_t indices_count, uint32_t indices[]);
SpvId spvb_vector_extract_dynamic(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId index);
SpvId spvb_vector_insert_dynamic(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId vector, SpvId component, SpvId index);
/// Used for almost all conversion operations
SpvId spvb_convert(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId target_type, SpvId value);
SpvId spvb_access_chain(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId element, size_t indices_count, SpvId indices[]);
SpvId spvb_ptr_access_chain(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId base, SpvId element, size_t indices_count, SpvId indices[]);
SpvId spvb_load(struct SpvBasicBlockBuilder* bb_builder, SpvId target_type, SpvId pointer, size_t operands_count, uint32_t operands[]);
void  spvb_store(struct SpvBasicBlockBuilder* bb_builder, SpvId value, SpvId pointer, size_t operands_count, uint32_t operands[]);
SpvId spvb_binop(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId result_type, SpvId lhs, SpvId rhs);
SpvId spvb_unop(struct SpvBasicBlockBuilder* bb_builder, SpvOp op, SpvId result_type, SpvId value);
SpvId spvb_elect(struct SpvBasicBlockBuilder* bb_builder, SpvId result_type, SpvId scope);
void  spvb_branch(struct SpvBasicBlockBuilder* bb_builder, SpvId target);
void  spvb_branch_conditional(struct SpvBasicBlockBuilder* bb_builder, SpvId condition, SpvId true_target, SpvId false_target);
void  spvb_switch(struct SpvBasicBlockBuilder* bb_builder, SpvId selector, SpvId default_target, size_t targets_count, SpvId* targets);
void  spvb_selection_merge(struct SpvBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvSelectionControlMask selection_control) ;
void  spvb_loop_merge(struct SpvBasicBlockBuilder* bb_builder, SpvId merge_bb, SpvId continue_bb, SpvLoopControlMask loop_control, size_t loop_control_ops_count, uint32_t loop_control_ops[]);
SpvId spvb_call(struct SpvBasicBlockBuilder* bb_builder, SpvId return_type, SpvId callee, size_t arguments_count, SpvId arguments[]);
SpvId spvb_ext_instruction(struct SpvBasicBlockBuilder* bb_builder, SpvId return_type, SpvId set, uint32_t instruction, size_t arguments_count, SpvId arguments[]);
void  spvb_return_void(struct SpvBasicBlockBuilder* bb_builder) ;
void  spvb_return_value(struct SpvBasicBlockBuilder* bb_builder, SpvId value);
void  spvb_unreachable(struct SpvBasicBlockBuilder* bb_builder);

SpvId spvb_parameter(struct SpvFnBuilder* fn_builder, SpvId param_type);
SpvId spvb_local_variable(struct SpvFnBuilder* fn_builder, SpvId type, SpvStorageClass storage_class);

void  spvb_name(struct SpvFileBuilder* file_builder, SpvId id, const char* str);

SpvId spvb_void_type(struct SpvFileBuilder* file_builder);
SpvId spvb_bool_type(struct SpvFileBuilder* file_builder);
SpvId spvb_int_type(struct SpvFileBuilder* file_builder, int width, bool signed_);
SpvId spvb_float_type(struct SpvFileBuilder* file_builder, int width);
SpvId spvb_ptr_type(struct SpvFileBuilder* file_builder, SpvStorageClass storage_class, SpvId element_type);
SpvId spvb_array_type(struct SpvFileBuilder* file_builder, SpvId element_type, SpvId dim);
SpvId spvb_runtime_array_type(struct SpvFileBuilder* file_builder, SpvId element_type);
SpvId spvb_fn_type(struct SpvFileBuilder* file_builder, size_t args_count, SpvId args_types[], SpvId codom);
SpvId spvb_struct_type(struct SpvFileBuilder* file_builder, size_t members_count, SpvId members[]);
SpvId spvb_vector_type(struct SpvFileBuilder* file_builder, SpvId component_type, uint32_t dim);
void spvb_bool_constant(struct SpvFileBuilder* file_builder, SpvId result, SpvId type, bool value);
void spvb_constant(struct SpvFileBuilder* file_builder, SpvId result, SpvId type, size_t bit_pattern_size, uint32_t bit_pattern[]);
SpvId spvb_constant_composite(struct SpvFileBuilder* file_builder, SpvId type, size_t ops_count, SpvId ops[]);
SpvId spvb_global_variable(struct SpvFileBuilder* file_builder, SpvId id, SpvId type, SpvStorageClass storage_class, bool has_initializer, SpvId initializer);

void  spvb_decorate(struct SpvFileBuilder* file_builder, SpvId target, SpvDecoration decoration, size_t extras_count, uint32_t extras[]);
void  spvb_decorate_member(struct SpvFileBuilder* file_builder, SpvId target, uint32_t member, SpvDecoration decoration, size_t extras_count, uint32_t extras[]);

SpvId spvb_debug_string(struct SpvFileBuilder* file_builder, const char* string);

void spvb_define_function(struct SpvFileBuilder* file_builder, struct SpvFnBuilder* fn_builder);

void  spvb_entry_point(struct SpvFileBuilder* file_builder, SpvExecutionModel execution_model, SpvId entry_point, const char* name, size_t interface_elements_count, SpvId interface_elements[]);
void  spvb_execution_mode(struct SpvFileBuilder* file_builder, SpvId entry_point, SpvExecutionMode execution_mode, size_t payloads_count, uint32_t payloads[]);
void  spvb_capability(struct SpvFileBuilder* file_builder, SpvCapability cap);

void  spvb_extension(struct SpvFileBuilder* file_builder, const char* name);

SpvId spvb_extended_import(struct SpvFileBuilder* file_builder, const char* name);

// SPV_KHR_shader_ballot
SpvId spvb_subgroup_ballot(struct SpvBasicBlockBuilder*, SpvId result_t, SpvId predicate);
SpvId spvb_subgroup_broadcast_first(struct SpvBasicBlockBuilder*, SpvId result_t, SpvId value);

struct SpvFileBuilder* spvb_begin();
void spvb_set_version(struct SpvFileBuilder* file_builder, uint8_t major, uint8_t minor);
void spvb_finish(struct SpvFileBuilder*, SpvSectionBuilder output);

struct SpvFnBuilder* spvb_begin_fn(struct SpvFileBuilder*, SpvId fn_id, SpvId fn_type, SpvId fn_ret_type);
struct SpvBasicBlockBuilder* spvb_begin_bb(struct SpvFileBuilder*, SpvId label);
void spvb_add_bb(struct SpvFnBuilder*, struct SpvBasicBlockBuilder*);

struct Phi* spvb_add_phi(struct SpvBasicBlockBuilder*, SpvId type, SpvId id);
void spvb_add_phi_source(struct Phi*, SpvId source_block, SpvId value);

SpvId get_block_builder_id(struct SpvBasicBlockBuilder*);

SpvId fn_ret_type_id(struct SpvFnBuilder*);

SpvId spvb_fresh_id(struct SpvFileBuilder* file_builder);

#endif
