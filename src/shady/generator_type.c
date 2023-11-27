#include "generator.h"

void generate(Growy* g, Data data) {
    generate_header(g, data);

    json_object* nodes = json_object_object_get(data.shd, "nodes");
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }

        json_object* t = json_object_object_get(node, "type");
        if (!t || json_object_get_boolean(t)) {
            json_object* ops = json_object_object_get(node, "ops");
            if (ops)
                growy_append_formatted(g, "const Type* check_type_%s(IrArena*, %s);\n", snake_name, name);
            else
                growy_append_formatted(g, "const Type* check_type_%s(IrArena*);\n", snake_name);
        }

        if (alloc)
            free(alloc);
    }
}
