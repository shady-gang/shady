target_sources(shady PRIVATE
        pipeline.c
        mem/memory.c
        cf/cf.c
        shader/shader_targets.c
        shader/specialize_execution_model.c
        polyfill/polyfill.c
)
