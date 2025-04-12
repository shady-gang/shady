#ifndef SHADY_SPIRV_BUILDER_H
#define SHADY_SPIRV_BUILDER_H

#include <spirv/unified1/spirv.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct SpvbBasicBlockBuilder_ SpvbBasicBlockBuilder;
typedef struct SpvbFnBuilder_ SpvbFnBuilder;
typedef struct SpvbFileBuilder_ SpvbFileBuilder;
typedef struct SpvbPhi_ SpvbPhi;

typedef struct Growy_ Growy;
typedef Growy* SpvbSectionBuilder;

typedef const char* String;

void spvb_literal_name(SpvbSectionBuilder data, const char* str);

SpvbFileBuilder* spvb_begin();
size_t spvb_finish(SpvbFileBuilder*, char** pwords);

SpvId spvb_fresh_id(SpvbFileBuilder*);

void spvb_set_version(SpvbFileBuilder*, uint8_t major, uint8_t minor);
void spvb_set_addressing_model(SpvbFileBuilder*, SpvAddressingModel model);
void spvb_capability(SpvbFileBuilder*, SpvCapability cap);
void spvb_extension(SpvbFileBuilder*, String name);
SpvId spvb_extended_import(SpvbFileBuilder*, String name);
void spvb_entry_point(SpvbFileBuilder*, SpvExecutionModel execution_model, SpvId entry_point, String name, size_t interface_elements_count, SpvId interface_elements[]);
void spvb_execution_mode(SpvbFileBuilder*, SpvId entry_point, SpvExecutionMode execution_mode, size_t payloads_count, uint32_t payloads[]);

// Debug info
SpvId spvb_debug_string(SpvbFileBuilder*, const char* string);
void spvb_name(SpvbFileBuilder*, SpvId id, const char* str);

// Decorations
void spvb_decorate(SpvbFileBuilder*, SpvId target, SpvDecoration decoration, size_t extras_count, uint32_t extras[]);
void spvb_decorate_member(SpvbFileBuilder*, SpvId target, uint32_t member, SpvDecoration decoration, size_t extras_count, uint32_t extras[]);

// Types
SpvId spvb_void_type(SpvbFileBuilder*);
SpvId spvb_bool_type(SpvbFileBuilder*);
SpvId spvb_int_type(SpvbFileBuilder*, int width, bool signed_);
SpvId spvb_float_type(SpvbFileBuilder*, int width);
SpvId spvb_forward_ptr_type(SpvbFileBuilder*, SpvStorageClass storage_class);
void spvb_ptr_type_define(SpvbFileBuilder*, SpvId id, SpvStorageClass storage_class, SpvId element_type);
SpvId spvb_ptr_type(SpvbFileBuilder*, SpvStorageClass storage_class, SpvId element_type);
SpvId spvb_array_type(SpvbFileBuilder*, SpvId element_type, SpvId dim);
SpvId spvb_runtime_array_type(SpvbFileBuilder*, SpvId element_type);
SpvId spvb_fn_type(SpvbFileBuilder*, size_t args_count, SpvId args_types[], SpvId codom);
SpvId spvb_struct_type(SpvbFileBuilder*, SpvId id, size_t members_count, SpvId members[]);
SpvId spvb_vector_type(SpvbFileBuilder*, SpvId component_type, uint32_t dim);
SpvId spvb_image_type(SpvbFileBuilder*, SpvId component_type, uint32_t dim, uint32_t depth, uint32_t onion, uint32_t multisample, uint32_t sampled, SpvImageFormat image_format);
SpvId spvb_sampler_type(SpvbFileBuilder*);
SpvId spvb_sampled_image_type(SpvbFileBuilder*, SpvId image_type);

// Constants and global variables
SpvId spvb_undef(SpvbFileBuilder*, SpvId type);
void spvb_bool_constant(SpvbFileBuilder*, SpvId result, SpvId type, bool value);
void spvb_constant(SpvbFileBuilder*, SpvId result, SpvId type, size_t bit_pattern_size, uint32_t bit_pattern[]);
SpvId spvb_constant_composite(SpvbFileBuilder*, SpvId type, size_t ops_count, SpvId ops[]);
SpvId spvb_constant_null(SpvbFileBuilder*, SpvId type);
SpvId spvb_global_variable(SpvbFileBuilder*, SpvId id, SpvId type, SpvStorageClass storage_class, bool has_initializer, SpvId initializer);
SpvId spvb_constant_op(SpvbFileBuilder*, SpvId type, SpvOp, size_t operands_count, SpvId operands[]);

// Function building stuff
SpvbFnBuilder* spvb_begin_fn(SpvbFileBuilder*, SpvId fn_id, SpvId fn_type, SpvId fn_ret_type);
SpvId spvb_fn_ret_type_id(SpvbFnBuilder* fnb);
SpvId spvb_parameter(SpvbFnBuilder* fn_builder, SpvId param_type);
SpvId spvb_local_variable(SpvbFnBuilder* fn_builder, SpvId type, SpvStorageClass storage_class);
void spvb_declare_function(SpvbFileBuilder*, SpvbFnBuilder* fn_builder);
void spvb_define_function(SpvbFileBuilder*, SpvbFnBuilder* fn_builder);

SpvbBasicBlockBuilder* spvb_begin_bb(SpvbFnBuilder*, SpvId label);
SpvbFnBuilder* spvb_get_fn_builder(SpvbBasicBlockBuilder*);
/// Actually adds the basic block to the function
/// This is a separate action from begin_bb because the ordering in which the basic blocks are written matters...
void spvb_add_bb(SpvbFnBuilder*, SpvbBasicBlockBuilder*);
SpvId spvb_get_block_builder_id(SpvbBasicBlockBuilder* basic_block_builder);

SpvbPhi* spvb_add_phi(SpvbBasicBlockBuilder*, SpvId type, SpvId id);
void spvb_add_phi_source(SpvbPhi*, SpvId source_block, SpvId value);
struct List* spvb_get_phis(SpvbBasicBlockBuilder* bb_builder);

// Normal instructions
SpvId spvb_op(SpvbBasicBlockBuilder*, SpvOp op, SpvId result_type, size_t operands_count, SpvId operands[]);
void spvb_no_result(SpvbBasicBlockBuilder*, SpvOp op, size_t operands_count, SpvId operands[]);
SpvId spvb_composite(SpvbBasicBlockBuilder*, SpvId aggregate_t, size_t elements_count, SpvId elements[]);
SpvId spvb_select(SpvbBasicBlockBuilder*, SpvId type, SpvId condition, SpvId if_true, SpvId if_false);
SpvId spvb_extract(SpvbBasicBlockBuilder*, SpvId target_type, SpvId composite, size_t indices_count, uint32_t indices[]);
SpvId spvb_insert(SpvbBasicBlockBuilder*, SpvId target_type, SpvId object, SpvId composite, size_t indices_count, uint32_t indices[]);
SpvId spvb_vector_extract_dynamic(SpvbBasicBlockBuilder*, SpvId target_type, SpvId vector, SpvId index);
SpvId spvb_vector_insert_dynamic(SpvbBasicBlockBuilder*, SpvId target_type, SpvId vector, SpvId component, SpvId index);
SpvId spvb_access_chain(SpvbBasicBlockBuilder*, SpvId target_type, SpvId element, size_t indices_count, SpvId indices[]);
SpvId spvb_ptr_access_chain(SpvbBasicBlockBuilder*, SpvId target_type, SpvId base, SpvId element, size_t indices_count, SpvId indices[]);
SpvId spvb_load(SpvbBasicBlockBuilder*, SpvId target_type, SpvId pointer, size_t operands_count, uint32_t operands[]);
void  spvb_store(SpvbBasicBlockBuilder*, SpvId value, SpvId pointer, size_t operands_count, uint32_t operands[]);
SpvId  spvb_vecshuffle(SpvbBasicBlockBuilder*, SpvId result_type, SpvId a, SpvId b, size_t operands_count, uint32_t operands[]);
SpvId spvb_group_elect(SpvbBasicBlockBuilder*, SpvId result_type, SpvId scope);
SpvId spvb_group_ballot(SpvbBasicBlockBuilder*, SpvId result_t, SpvId predicate, SpvId scope);
SpvId spvb_group_shuffle(SpvbBasicBlockBuilder*, SpvId result_type, SpvId scope, SpvId value, SpvId id);
SpvId spvb_group_broadcast_first(SpvbBasicBlockBuilder*, SpvId result_t, SpvId value, SpvId scope);
SpvId spvb_group_non_uniform_group_op(SpvbBasicBlockBuilder*, SpvId result_t, SpvOp op, SpvId scope, SpvGroupOperation group_op, SpvId value, SpvId* cluster_size);

// Terminators
void  spvb_branch(SpvbBasicBlockBuilder*, SpvId target);
void  spvb_branch_conditional(SpvbBasicBlockBuilder*, SpvId condition, SpvId true_target, SpvId false_target);
void  spvb_switch(SpvbBasicBlockBuilder*, SpvId selector, SpvId default_target, size_t targets_count, SpvId* targets);
void  spvb_selection_merge(SpvbBasicBlockBuilder*, SpvId merge_bb, SpvSelectionControlMask selection_control) ;
void  spvb_loop_merge(SpvbBasicBlockBuilder*, SpvId merge_bb, SpvId continue_bb, SpvLoopControlMask loop_control, size_t loop_control_ops_count, uint32_t loop_control_ops[]);
SpvId spvb_call(SpvbBasicBlockBuilder*, SpvId return_type, SpvId callee, size_t arguments_count, SpvId arguments[]);
SpvId spvb_ext_instruction(SpvbBasicBlockBuilder*, SpvId return_type, SpvId set, uint32_t instruction, size_t arguments_count, SpvId arguments[]);
void  spvb_return_void(SpvbBasicBlockBuilder*) ;
void  spvb_return_value(SpvbBasicBlockBuilder*, SpvId value);
void  spvb_unreachable(SpvbBasicBlockBuilder*);
void spvb_terminator(SpvbBasicBlockBuilder*, SpvOp, size_t operands_count, SpvId operands[]);

#endif
