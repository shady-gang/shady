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
    ShdIntSize ptr_size;

    struct {
        bool physical;
        bool allowed;
    } address_spaces[NumAddressSpaces];
} PtrModel;

PtrModel get_full_physical_ptr_model(ShdIntSize ptr_size);

typedef struct {
    /// The base type for emulated memory
    ShdIntSize word_size;
    ShdIntSize fn_ptr_size;
    /// Minimum alignment
    uint64_t min_align;
} MemoryLayoutRules;

typedef struct {
    ShdScope constants;
    ShdScope gang;
    ShdScope bottom;
} ScopesLattice;

ScopesLattice get_default_scopes_lattice(void);

typedef enum {
    TgtNone,
    TgtSPV,
    TgtC,
    TgtGLSL,
    TgtISPC,
    TgtCUDA,
} CodegenTarget;

typedef struct {
    CodegenTarget arch;
    PtrModel ptr_model;

    ShdIntSize fn_ptr_size;
    uint32_t subgroup_size;

    struct {
        bool native_stack;
        bool native_memcpy;
        bool native_fncalls;
        bool native_tailcalls;
        //bool rt_pipelines;
        bool linkage;
        bool maximal_reconvergence;
    } capabilities;

    //const ExecutionModelInfo* exec_info;
    //ShdExecutionModel execution_model;
    //String entry_point;
} TargetConfig;

TargetConfig shd_default_target_config(void);

/// Changes various fields in the target to be accurate defaults for the target arch
void shd_target_configure_defaults_for_arch(TargetConfig*);

/// Removes some capabilities in the target according to the chosen execution model
void shd_target_apply_execution_model_restrictions(TargetConfig*, ShdExecutionModel);

/// Some useful properties of the target machine are enforced at the IR level
/// But they can also be altered over time, so we decouple these from the actual TargetConfig
typedef struct {
    PtrModel ptr;
    MemoryLayoutRules memory;
    ScopesLattice scopes;
    ShdIntSize exec_mask_size;
} MachineRules;

MachineRules get_machine_rules_from_target_config(const TargetConfig* target);

typedef struct ArenaConfig_ ArenaConfig;
struct ArenaConfig_ {
    bool name_bound;
    bool check_op_classes;
    bool check_types;
    bool allow_fold;
    bool validate_builtin_types; // do @Builtins variables need to match their type in builtins.h ?

    MachineRules rules;

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

ArenaConfig shd_default_arena_config(const MachineRules* target);
const ArenaConfig* shd_get_arena_config(const IrArena* a);

static inline const Type* shd_uword_type(IrArena* a) {
    return int_type_helper(a, shd_get_arena_config(a)->rules.memory.word_size, 0);
}

static inline const Type* shd_usize_type(IrArena* a) {
    return int_type_helper(a, shd_get_arena_config(a)->rules.ptr.ptr_size, 0);
}

typedef struct CompilerConfig_ CompilerConfig;
struct CompilerConfig_ {
    struct {
        bool emulate_subgroup_ops;
        bool emulate_subgroup_ops_extended_types;
        bool int64;
        bool decay_ptrs;
        bool use_scratch_for_private;
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
        bool max_stack_size;
        bool scratch_base_addr;
    } printf_trace;

    struct {
        int max_top_iterations;
    } shader_diagnostics;

    // struct {
    //     struct { void* uptr; void (*fn)(void*, String, Module*); } after_pass;
    // } hooks;
};

CompilerConfig shd_default_compiler_config(void);

#endif
