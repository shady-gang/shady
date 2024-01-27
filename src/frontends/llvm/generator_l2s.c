#include "generator.h"

void generate_llvm_shady_address_space_conversion(Growy* g, json_object* address_spaces) {
    growy_append_formatted(g, "AddressSpace convert_llvm_address_space(unsigned as) {\n");
    growy_append_formatted(g, "\tstatic bool warned = false;\n");
    growy_append_formatted(g, "\tswitch (as) {\n");
    for (size_t i = 0; i < json_object_array_length(address_spaces); i++) {
        json_object* as = json_object_array_get_idx(address_spaces, i);
        String name = json_object_get_string(json_object_object_get(as, "name"));
        json_object* llvm_id = json_object_object_get(as, "llvm-id");
        if (!llvm_id || json_object_get_type(llvm_id) != json_type_int)
            continue;
        growy_append_formatted(g, "\t\t case %d: return As%s;\n", json_object_get_int(llvm_id), name);
    }
    growy_append_formatted(g, "\t\tdefault:\n");
    growy_append_formatted(g, "\t\t\tif (!warned)\n");
    growy_append_string(g, "\t\t\t\twarn_print(\"Warning: unrecognised address space %d\", as);\n");
    growy_append_formatted(g, "\t\t\twarned = true;\n");
    growy_append_formatted(g, "\t\t\treturn AsGeneric;\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "}\n");
}

void generate(Growy* g, json_object* src) {
    generate_header(g, src);
    growy_append_formatted(g, "#include \"l2s_private.h\"\n");
    growy_append_formatted(g, "#include \"log.h\"\n");
    growy_append_formatted(g, "#include <stdbool.h>\n");

    generate_llvm_shady_address_space_conversion(g, json_object_object_get(src, "address-spaces"));
}
