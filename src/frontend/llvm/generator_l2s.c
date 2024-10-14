#include "generator.h"

void generate_llvm_shady_address_space_conversion(Growy* g, json_object* address_spaces) {
    shd_growy_append_formatted(g, "AddressSpace l2s_convert_llvm_address_space(unsigned as) {\n");
    shd_growy_append_formatted(g, "\tstatic bool warned = false;\n");
    shd_growy_append_formatted(g, "\tswitch (as) {\n");
    for (size_t i = 0; i < json_object_array_length(address_spaces); i++) {
        json_object* as = json_object_array_get_idx(address_spaces, i);
        String name = json_object_get_string(json_object_object_get(as, "name"));
        json_object* llvm_id = json_object_object_get(as, "llvm-id");
        if (!llvm_id || json_object_get_type(llvm_id) != json_type_int)
            continue;
        shd_growy_append_formatted(g, "\t\t case %d: return As%s;\n", json_object_get_int(llvm_id), name);
    }
    shd_growy_append_formatted(g, "\t\tdefault:\n");
    shd_growy_append_formatted(g, "\t\t\tif (!warned)\n");
    shd_growy_append_string(g, "\t\t\t\tshd_warn_print(\"Warning: unrecognised address space %d\", as);\n");
    shd_growy_append_formatted(g, "\t\t\twarned = true;\n");
    shd_growy_append_formatted(g, "\t\t\treturn AsGeneric;\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "}\n");
}

void generate(Growy* g, json_object* src) {
    generate_header(g, src);
    shd_growy_append_formatted(g, "#include \"l2s_private.h\"\n");
    shd_growy_append_formatted(g, "#include \"log.h\"\n");
    shd_growy_append_formatted(g, "#include <stdbool.h>\n");

    generate_llvm_shady_address_space_conversion(g, json_object_object_get(src, "address-spaces"));
}
