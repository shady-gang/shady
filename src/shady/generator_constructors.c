#include "generator.h"

static void generate_pre_construction_validation(Growy* g, Data data) {
    json_object* nodes = json_object_object_get(data.shd, "nodes");
    growy_append_formatted(g, "void pre_construction_validation(IrArena* arena, Node* node) {\n");
    growy_append_formatted(g, "\tswitch (node->tag) { \n");
    assert(json_object_get_type(nodes) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }
        growy_append_formatted(g, "\tcase %s_TAG: {\n", name);
        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                String class = json_object_get_string(json_object_object_get(op, "class"));
                if (!class)
                    continue;
                bool list = json_object_get_boolean(json_object_object_get(op, "list"));
                if (strcmp(class, "string") == 0) {
                    if (!list)
                        growy_append_formatted(g, "\t\tnode->payload.%s.%s = string(arena, node->payload.%s.%s);\n", snake_name, op_name, snake_name, op_name);
                    else
                        growy_append_formatted(g, "\t\tnode->payload.%s.%s = import_strings(arena, node->payload.%s.%s);\n", snake_name, op_name, snake_name, op_name);
                } else {
                    String cap = capitalize(class);
                    growy_append_formatted(g, "\t\t{\n");
                    String extra = "";
                    if (list) {
                        growy_append_formatted(g, "\t\t\tNodes ops = node->payload.%s.%s;\n", snake_name, op_name);
                        growy_append_formatted(g, "\t\t\tfor (size_t i = 0; i < ops.count; i++) {\n");
                        growy_append_formatted(g, "\t\t\tconst Node* op = ops.nodes[i];\n");
                        extra = "\t";
                    }
                    if (!list)
                        growy_append_formatted(g, "\t\t\tconst Node* op = node->payload.%s.%s;\n", snake_name, op_name);
                    growy_append_formatted(g, "%s\t\t\tif (arena->config.check_op_classes && op != NULL && !is_%s(op)) {\n", extra, class);
                    growy_append_formatted(g, "%s\t\t\t\terror_print(\"Invalid '%s' operand for node '%s', expected a %s\");\n", extra, op_name, name, class);
                    growy_append_formatted(g, "%s\t\t\t\terror_die();\n", extra);
                    growy_append_formatted(g, "%s\t\t\t}\n", extra);
                    if (list)
                        growy_append_formatted(g, "\t\t\t}\n");
                    free(cap);
                    growy_append_formatted(g, "\t\t}\n");
                }
            }
        }
        growy_append_formatted(g, "\t\tbreak;\n");
        growy_append_formatted(g, "\t}\n", name);
        if (alloc)
            free(alloc);
    }
    growy_append_formatted(g, "\t\tdefault: break;\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "}\n\n");
}

void generate(Growy* g, Data data) {
    generate_header(g, data);

    json_object* nodes = json_object_object_get(data.shd, "nodes");
    generate_node_ctor(g, nodes, true);
    generate_pre_construction_validation(g, data);
}
