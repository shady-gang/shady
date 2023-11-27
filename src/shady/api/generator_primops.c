#include "generator.h"

void generate(Growy* g, Data data) {
    generate_header(g, data);

    json_object* nodes = json_object_object_get(data.shd, "prim-ops");
    growy_append_formatted(g, "typedef enum Op_ {\n");

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        growy_append_formatted(g, "\t%s_op,\n", name);
    }

    growy_append_formatted(g, "\tPRIMOPS_COUNT,\n");
    growy_append_formatted(g, "} Op;\n");
}
