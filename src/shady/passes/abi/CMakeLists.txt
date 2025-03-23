target_sources(shady PRIVATE
        call_init_fini.c
        lower_generic_globals.c
        lower_workgroups.c
        specialize_entry_point.c
        specialize_execution_model.c
)