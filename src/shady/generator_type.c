#include "generator.h"

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    json_object* nodes = json_object_object_get(src, "nodes");

    shd_growy_append_formatted(g, "const Type* check_type_generated(IrArena* a, const Node* node) {\n");
    shd_growy_append_formatted(g, "\tswitch(node->tag) {\n");
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            snake_name = to_snake_case(name);
            alloc = (void*) snake_name;
        }

        json_object* t = json_object_object_get(node, "type");
        if (!t || json_object_get_boolean(t)) {
            shd_growy_append_formatted(g, "\t\tcase %s_TAG: ", name);
            json_object* ops = json_object_object_get(node, "ops");
            if (ops)
                shd_growy_append_formatted(g, "return check_type_%s(a, node->payload.%s);\n", snake_name, snake_name);
            else
                shd_growy_append_formatted(g, "return check_type_%s(a);\n", snake_name);
        }

        if (alloc)
            free(alloc);
    }
    shd_growy_append_formatted(g, "\t\tdefault: return NULL;\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "}\n");
}
