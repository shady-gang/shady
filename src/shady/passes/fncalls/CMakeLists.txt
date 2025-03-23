embed_file(string shady_scheduler_src scheduler.slim)
target_link_libraries(shady PRIVATE "$<BUILD_INTERFACE:shady_scheduler_src>")

target_sources(shady PRIVATE
        lift_indirect_targets.c
        lower_callf.c
        lower_dynamic_control.c
        lower_tailcalls.c
        mark_leaf_functions.c
        scheduler.c
)