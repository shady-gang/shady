#ifndef SHD_CONFIG_H
#define SHD_CONFIG_H

#include "shady/ir/base.h"
#include "shady/ir/int.h"
#include "shady/ir/grammar.h"
#include "shady/ir/builtin.h"
#include "shady/ir/execution_model.h"

typedef enum {
    ShdFeatureSupportBanned,
    ShdFeatureSupportSupported,
    ShdFeatureSupportEmulate,
} ShdFeatureSupport;

typedef struct {
    IntSizes ptr_size;
    /// The base type for emulated memory
    IntSizes word_size;

    IntSizes fn_ptr_size;
    IntSizes exec_mask_size;

    struct {
        bool physical;
        bool allowed;
    } address_spaces[NumAddressSpaces];
} MemoryModel;

typedef enum {
    /// Uses the MaskType
    SubgroupMaskAbstract,
    /// Uses a 64-bit integer
    SubgroupMaskInt64
} SubgroupMaskRepresentation;

typedef struct {
    MemoryModel memory;

    struct {
        ShdScope constants;
        ShdScope gang;
        ShdScope bottom;
    } scopes;

    struct {
        bool native_tailcalls;
    } capabilities;

    /// Selects which type the subgroup intrinsic primops use to manipulate masks
    SubgroupMaskRepresentation subgroup_mask_representation;

    ExecutionModel execution_model;
    uint32_t subgroup_size;

} TargetConfig;

TargetConfig shd_default_target_config(void);

typedef struct ArenaConfig_ ArenaConfig;
struct ArenaConfig_ {
    bool name_bound;
    bool check_op_classes;
    bool check_types;
    bool allow_fold;
    bool validate_builtin_types; // do @Builtins variables need to match their type in builtins.h ?

    struct {
        uint32_t workgroup_size[3];
    } specializations;

    TargetConfig target;

    /// 'folding' optimisations - happen in the constructors directly
    struct {
        bool inline_single_use_bbs;
        bool fold_static_control_flow;
        bool delete_unreachable_structured_cases;
        bool weaken_non_leaking_allocas;
        bool weaken_bitcast_to_lea;
        bool assume_fixed_memory_layout;
    } optimisations;
};

ArenaConfig shd_default_arena_config(const TargetConfig* target);
const ArenaConfig* shd_get_arena_config(const IrArena* a);

typedef struct CompilerConfig_ CompilerConfig;
struct CompilerConfig_ {
    bool dynamic_scheduling;
    uint32_t per_thread_stack_size;

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
        bool top_function;
        bool stack_size;
        bool subgroup_ops;
    } printf_trace;

    struct {
        int max_top_iterations;
    } shader_diagnostics;

    struct {
        String entry_point;
        ExecutionModel execution_model;
    } specialization;

    TargetConfig target;

    // struct {
    //     struct { void* uptr; void (*fn)(void*, String, Module*); } after_pass;
    // } hooks;
};

CompilerConfig shd_default_compiler_config(void);

#endif
