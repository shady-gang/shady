#include "generator.h"

bool has_custom_ctor(json_object* node) {
    String constructor = json_object_get_string(json_object_object_get(node, "constructor"));
    return (constructor && strcmp(constructor, "custom") == 0);
}

void generate_node_ctor(Growy* g, json_object* nodes, bool definition) {
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        if (has_custom_ctor(node))
            continue;

        if (definition && i > 0)
            growy_append_formatted(g, "\n");

        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }

        String ap = definition ? " arena" : "";
        json_object* ops = json_object_object_get(node, "ops");
        if (ops)
            growy_append_formatted(g, "const Node* %s(IrArena*%s, %s%s)", snake_name, ap, name, definition ? " payload" : "");
        else
            growy_append_formatted(g, "const Node* %s(IrArena*%s)", snake_name, ap);

        if (definition) {
            growy_append_formatted(g, " {\n");
            growy_append_formatted(g, "\tNode node;\n");
            growy_append_formatted(g, "\tmemset((void*) &node, 0, sizeof(Node));\n");
            growy_append_formatted(g, "\tnode = (Node) {\n");
            growy_append_formatted(g, "\t\t.arena = arena,\n");
            growy_append_formatted(g, "\t\t.tag = %s_TAG,\n", name);
            if (ops)
                growy_append_formatted(g, "\t\t.payload.%s = payload,\n", snake_name);
            json_object* t = json_object_object_get(node, "type");
            if (!t || json_object_get_boolean(t)) {
                growy_append_formatted(g, "\t\t.type = arena->config.check_types ? ");
                if (ops)
                    growy_append_formatted(g, "check_type_%s(arena, payload)", snake_name);
                else
                    growy_append_formatted(g, "check_type_%s(arena)", snake_name);
                growy_append_formatted(g, ": NULL,\n");
            } else
                growy_append_formatted(g, "\t\t.type = NULL,\n");
            growy_append_formatted(g, "\t};\n");
            growy_append_formatted(g, "\treturn create_node_helper(arena, node, NULL);\n");
            growy_append_formatted(g, "}\n");
        } else {
            growy_append_formatted(g, ";\n");
        }

        if (alloc)
            free(alloc);
    }
    growy_append_formatted(g, "\n");
}