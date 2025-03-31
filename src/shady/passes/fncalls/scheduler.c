#include "../../../frontend/slim/parser.h"
#include "shady_scheduler_src.h"

#include "printer.h"
#include "log.h"

#include <stdlib.h>

void shd_add_scheduler_source(const CompilerConfig* config, Module* dst) {
    SlimParserConfig pconfig = {
            .front_end = true,
            .target_config = &config->target
    };
    Printer* p = shd_new_printer_from_growy(shd_new_growy());
    shd_print(p, "@Internal @Alias type fn_ptr_t = u%d;\n", int_size_in_bytes(config->target.memory.fn_ptr_size) * 8);
    // SUBGROUPS_PER_WG = (NUMBER OF INVOCATIONS IN SUBGROUP / SUBGROUP SIZE)
    // Note: this computations assumes only full subgroups are launched, if subgroups can launch partially filled then this relationship does not hold.
    uint32_t wg_size[3];
    wg_size[0] = shd_get_arena_config(shd_module_get_arena(dst))->specializations.workgroup_size[0];
    wg_size[1] = shd_get_arena_config(shd_module_get_arena(dst))->specializations.workgroup_size[1];
    wg_size[2] = shd_get_arena_config(shd_module_get_arena(dst))->specializations.workgroup_size[2];
    uint32_t subgroups_per_wg = (wg_size[0] * wg_size[1] * wg_size[2]) / config->target.subgroup_size;
    if (subgroups_per_wg == 0)
        subgroups_per_wg = 1; // uh-oh
    shd_print(p, "@Exported @Internal const u32 SUBGROUPS_PER_WG = %d;\n", subgroups_per_wg);
    shd_print(p, "%s", shady_scheduler_src);
    String s = shd_printer_growy_unwrap(p);
    Module* builtin_scheduler_mod = shd_parse_slim_module(config, &pconfig, s, "builtin_scheduler");
    free((char*) s);
    shd_debug_print("Adding builtin scheduler code");
    shd_module_link(dst, builtin_scheduler_mod);
    shd_destroy_ir_arena(shd_module_get_arena(builtin_scheduler_mod));
}