#ifndef SHD_CONFIG_H
#define SHD_CONFIG_H

#include "shady/ir/base.h"
#include "shady/ir/int.h"
#include "shady/ir/grammar.h"
#include "shady/ir/execution_model.h"

typedef struct {
    IntSizes ptr_size;
    /// The base type for emulated memory
    IntSizes word_size;
} PointerModel;

typedef struct {
    PointerModel memory;
} TargetConfig;

TargetConfig shd_default_target_config(void);

typedef enum {
    /// Uses the MaskType
    SubgroupMaskAbstract,
    /// Uses a 64-bit integer
    SubgroupMaskInt64
} SubgroupMaskRepresentation;

typedef struct ArenaConfig_ ArenaConfig;
struct ArenaConfig_ {
    bool name_bound;
    bool check_op_classes;
    bool check_types;
    bool allow_fold;
    bool validate_builtin_types; // do @Builtins variables need to match their type in builtins.h ?
    bool is_simt;

    struct {
        bool physical;
        bool allowed;
    } address_spaces[NumAddressSpaces];

    struct {
        /// Selects which type the subgroup intrinsic primops use to manipulate masks
        SubgroupMaskRepresentation subgroup_mask_representation;

        uint32_t workgroup_size[3];
    } specializations;

    PointerModel memory;

    /// 'folding' optimisations - happen in the constructors directly
    struct {
        bool inline_single_use_bbs;
        bool fold_static_control_flow;
        bool delete_unreachable_structured_cases;
        bool weaken_non_leaking_allocas;
    } optimisations;
};

ArenaConfig shd_default_arena_config(const TargetConfig* target);
const ArenaConfig* shd_get_arena_config(const IrArena* a);

typedef struct CompilerConfig_ CompilerConfig;
struct CompilerConfig_ {
    bool dynamic_scheduling;
    uint32_t per_thread_stack_size;

    struct {
        uint8_t major;
        uint8_t minor;
    } target_spirv_version;

    struct {
        bool restructure_with_heuristics;
        bool add_scope_annotations;
        bool has_scope_annotations;
    } input_cf;

    struct {
        bool emulate_generic_ptrs;
        bool emulate_physical_memory;

        bool emulate_subgroup_ops;
        bool emulate_subgroup_ops_extended_types;
        bool int64;
        bool decay_ptrs;
    } lower;

    struct {
        bool spv_shuffle_instead_of_broadcast_first;
        bool force_join_point_lifting;
    } hacks;

    struct {
        struct {
            bool after_every_pass;
            bool delete_unused_instructions;
        } cleanup;
        bool inline_everything;
    } optimisations;

    struct {
        bool memory_accesses;
        bool stack_accesses;
        bool god_function;
        bool stack_size;
        bool subgroup_ops;
    } printf_trace;

    struct {
        int max_top_iterations;
    } shader_diagnostics;

    struct {
        bool print_generated, print_builtin, print_internal;
    } logging;

    struct {
        String entry_point;
        ExecutionModel execution_model;
        uint32_t subgroup_size;
    } specialization;

    TargetConfig target;

    struct {
        struct { void* uptr; void (*fn)(void*, String, Module*); } after_pass;
    } hooks;
};

CompilerConfig shd_default_compiler_config(void);

#endif
