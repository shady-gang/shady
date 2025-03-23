target_sources(shady PRIVATE
        lift_indirect_targets.c
        lower_callf.c
        lower_dynamic_control.c
        lower_tailcalls.c
        mark_leaf_functions.c
)