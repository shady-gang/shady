execute_process(COMMAND ${VCC} ${SRC}/${T} ${TARGS} -o ${T}.spv COMMAND_ERROR_IS_FATAL ANY)

find_program(SPIRV_VALIDATOR "spirv-val")
if (SPIRV_VALIDATOR)
    message("Validating stuff ${SPIRV_VALIDATOR}")
    execute_process(COMMAND ${SPIRV_VALIDATOR} ${T}.spv --target-env vulkan1.3 COMMAND_ERROR_IS_FATAL ANY)
endif ()