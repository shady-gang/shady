#include "generator.h"

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    json_object* nodes = json_object_object_get(src, "prim-ops");
    shd_growy_append_formatted(g, "typedef enum Op_ {\n");

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        shd_growy_append_formatted(g, "\t%s_op,\n", name);
    }

    shd_growy_append_formatted(g, "\tPRIMOPS_COUNT,\n");
    shd_growy_append_formatted(g, "} Op;\n");

    json_object* op_classes = json_object_object_get(src, "prim-ops-classes");
    generate_bit_enum(g, "OpClass", "Oc", op_classes);
    shd_growy_append_formatted(g, "OpClass get_primop_class(Op);\n\n");
}
