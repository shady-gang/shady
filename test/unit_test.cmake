if (VALSPV)
    list(APPEND COMP_ARGS -o ${DST}/${NAME}.spv)
endif()

execute_process(COMMAND ${COMPILER} ${FILES} ${COMP_ARGS} COMMAND_ERROR_IS_FATAL ANY COMMAND_ECHO STDOUT)

if (VALSPV)
    #find_program(SPIRV_VALIDATOR "spirv-val")
    if (SPIRV_VALIDATOR)
        #message("Validating stuff ${SPIRV_VALIDATOR} (args) ${SPV_VAL_ARGS}")
        execute_process(COMMAND ${SPIRV_VALIDATOR} ${DST}/${NAME}.spv ${SPV_VAL_ARGS} COMMAND_ERROR_IS_FATAL ANY)
    endif ()
endif ()