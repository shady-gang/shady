#include "shady/ir.h"
#include "passes/passes.h"
#include "log.h"
#include "portability.h"
#include "analysis/verify.h"
#include "parser/parser.h"
#include "builtin_code.h"

#define KiB * 1024
#define MiB * 1024 KiB

CompilerConfig default_compiler_config() {
    return (CompilerConfig) {
        .allow_frontend_syntax = false,
        .per_thread_stack_size = 32 KiB,
        .per_subgroup_stack_size = 1 KiB
    };
}

#define RUN_PASS(pass_name) \
old_arena = *arena;                                        \
*arena = new_arena(aconfig);                               \
*program = pass_name(config, old_arena, *arena, *program); \
info_print("After "#pass_name" pass: \n");                 \
info_node(*program);                                       \
verify_program(*program);                                  \
destroy_arena(old_arena);

CompilationResult run_compiler_passes(SHADY_UNUSED CompilerConfig* config, IrArena** arena, const Node** program) {
    ArenaConfig aconfig = {
        .allow_fold = false,
        .check_types = false
    };
    IrArena* old_arena;

    RUN_PASS(bind_program)
    RUN_PASS(normalize)

    aconfig.check_types = true;
    RUN_PASS(infer_program)

    aconfig.allow_fold = true;

    RUN_PASS(lower_cf_instrs)
    RUN_PASS(lower_callc)
    RUN_PASS(lower_callf)
    RUN_PASS(lower_tailcalls)

    aconfig.subgroup_mask_representation = SubgroupMaskSpvKHRBallot;
    RUN_PASS(lower_mask)
    RUN_PASS(lower_stack)
    RUN_PASS(lower_physical_ptrs)

    return CompilationNoError;
}

static size_t num_builtin_sources_files = 1;
static const char* builtin_source_files[] = { builtin_scheduler_txt };

CompilationResult parse_files(CompilerConfig* config, size_t num_files, const char** files_contents, IrArena* arena, const Node** program) {
    size_t num_source_files = num_builtin_sources_files + num_files;

    LARRAY(const Node*, parsed_files, num_source_files);
    size_t total_decls_count = 0;
    for (size_t i = 0; i < num_source_files; i++) {
        const char* input_file_contents = NULL;

        if (i < num_builtin_sources_files) {
            input_file_contents = builtin_source_files[i];
        } else {
            input_file_contents = files_contents[i - num_builtin_sources_files];
        }

        info_print("Parsing: \n%s\n", input_file_contents);
        ParserConfig pconfig = {
            .front_end = config->allow_frontend_syntax
        };
        const Node* parsed_file = parse(pconfig, input_file_contents, arena);
        parsed_files[i] = parsed_file;
        assert(parsed_file->tag == Root_TAG);
        total_decls_count += parsed_file->payload.root.declarations.count;
    }

    // Merge all declarations into a giant program
    const Node** all_decls = malloc(sizeof(const Node*) * total_decls_count);
    size_t num_decl = 0;
    for (size_t i = 0; i < num_source_files; i++) {
        const Node* parsed_file = parsed_files[i];
        for (size_t j = 0; j < parsed_file->payload.root.declarations.count; j++)
            all_decls[num_decl++] = parsed_file->payload.root.declarations.nodes[j];
    }
    *program = root(arena, (Root) {
        .declarations = nodes(arena, num_decl, all_decls)
    });
    free(all_decls);
    return CompilationNoError;
}
