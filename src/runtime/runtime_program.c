#include "runtime_private.h"
#include "shady/driver.h"

#include "log.h"
#include "list.h"
#include "util.h"
#include "portability.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../shady/transform/memory_layout.h"

Program* new_program_from_module(Runtime* runtime, const CompilerConfig* base_config, Module* mod) {
    Program* program = calloc(1, sizeof(Program));
    program->runtime = runtime;
    program->base_config = base_config;
    program->arena = NULL;
    program->module = mod;

    // TODO split the compilation pipeline into generic and non-generic parts
    append_list(Program*, runtime->programs, program);
    return program;
}

Program* load_program(Runtime* runtime, const CompilerConfig* base_config, const char* program_src) {
    IrArena* arena = new_ir_arena(default_arena_config());
    Module* module = new_module(arena, "my_module");

    int err = driver_load_source_file(SrcShadyIR, strlen(program_src), program_src, module);
    if (err != NoError) {
        return NULL;
    }

    Program* program = new_program_from_module(runtime, base_config, module);
    program->arena = arena;
    return program;
}

Program* load_program_from_disk(Runtime* runtime, const CompilerConfig* base_config, const char* path) {
    IrArena* arena = new_ir_arena(default_arena_config());
    Module* module = new_module(arena, "my_module");

    int err = driver_load_source_file_from_filename(path, module);
    if (err != NoError) {
        return NULL;
    }

    Program* program = new_program_from_module(runtime, base_config, module);
    program->arena = arena;
    return program;
}

void unload_program(Program* program) {
    // TODO iterate over the specialized stuff
    if (program->arena) // if the program owns an arena
        destroy_ir_arena(program->arena);
    free(program);
}

bool shd_extract_parameters_info(ProgramParamsInfo* parameters, Module* mod) {
    Nodes decls = get_module_declarations(mod);

    const Node* args_struct_annotation;
    const Node* args_struct_type = NULL;
    const Node* entry_point_function = NULL;

    for (int i = 0; i < decls.count; ++i) {
        const Node* node = decls.nodes[i];

        switch (node->tag) {
            case GlobalVariable_TAG: {
                const Node* entry_point_args_annotation = lookup_annotation(node, "EntryPointArgs");
                if (entry_point_args_annotation) {
                    if (node->payload.global_variable.type->tag != RecordType_TAG) {
                        error_print("EntryPointArgs must be a struct\n");
                        return false;
                    }

                    if (args_struct_type) {
                        error_print("there cannot be more than one EntryPointArgs\n");
                        return false;
                    }

                    args_struct_annotation = entry_point_args_annotation;
                    args_struct_type = node->payload.global_variable.type;
                }
                break;
            }
            case Function_TAG: {
                if (lookup_annotation(node, "EntryPoint")) {
                    if (node->payload.fun.params.count != 0) {
                        error_print("EntryPoint cannot have parameters\n");
                        return false;
                    }

                    if (entry_point_function) {
                        error_print("there cannot be more than one EntryPoint\n");
                        return false;
                    }

                    entry_point_function = node;
                }
                break;
            }
            default: break;
        }
    }

    if (!entry_point_function) {
        error_print("could not find EntryPoint\n");
        return false;
    }

    if (!args_struct_type) {
        *parameters = (ProgramParamsInfo) { .num_args = 0 };
        return true;
    }

    if (args_struct_annotation->tag != AnnotationValue_TAG) {
        error_print("EntryPointArgs annotation must contain exactly one value\n");
        return false;
    }

    const Node* annotation_fn = args_struct_annotation->payload.annotation_value.value;
    assert(annotation_fn->tag == FnAddr_TAG);
    if (annotation_fn->payload.fn_addr.fn != entry_point_function) {
        error_print("EntryPointArgs annotation refers to different EntryPoint\n");
        return false;
    }

    size_t num_args = args_struct_type->payload.record_type.members.count;

    if (num_args == 0) {
        error_print("EntryPointArgs cannot be empty\n");
        return false;
    }

    IrArena* a = get_module_arena(mod);

    LARRAY(FieldLayout, fields, num_args);
    get_record_layout(a, args_struct_type, fields);

    size_t* offset_size_buffer = calloc(1, 2 * num_args * sizeof(size_t));
    if (!offset_size_buffer) {
        error_print("failed to allocate EntryPointArgs offsets and sizes array\n");
        return false;
    }
    size_t* offsets = offset_size_buffer;
    size_t* sizes = offset_size_buffer + num_args;

    for (int i = 0; i < num_args; ++i) {
        offsets[i] = fields[i].offset_in_bytes;
        sizes[i] = fields[i].mem_layout.size_in_bytes;
    }

    parameters->num_args = num_args;
    parameters->arg_offset = offsets;
    parameters->arg_size = sizes;
    parameters->args_size = offsets[num_args - 1] + sizes[num_args - 1];
    return true;
}
