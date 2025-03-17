execute_process(COMMAND ${COMPILER} ${FILES} ${TARGS} -o ${DST}/${T}.spv COMMAND_ERROR_IS_FATAL ANY COMMAND_ECHO STDOUT)

find_program(SPIRV_VALIDATOR "spirv-val")
if (SPIRV_VALIDATOR)
    message("Validating stuff ${SPIRV_VALIDATOR}")
    execute_process(COMMAND ${SPIRV_VALIDATOR} ${DST}/${T}.spv --target-env vulkan1.3 COMMAND_ERROR_IS_FATAL ANY)
endif ()