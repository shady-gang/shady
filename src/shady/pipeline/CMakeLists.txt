target_sources(shady PRIVATE
        pipeline.c
        mem/memory.c
        cf/cf.c
        lower/fncalls/lower_dynamic_control.c
        shader/shader_targets.c
        shader/specialize_execution_model.c
        polyfill/polyfill.c
        mem/lower_top_level_globals.c
)
