#include "generator.h"

static void generate_pre_construction_validation(Growy* g, json_object* src) {
    json_object* nodes = json_object_object_get(src, "nodes");
    shd_growy_append_formatted(g, "void pre_construction_validation(IrArena* arena, Node* node) {\n");
    shd_growy_append_formatted(g, "\tswitch (node->tag) { \n");
    assert(json_object_get_type(nodes) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            snake_name = to_snake_case(name);
            alloc = (void*) snake_name;
        }
        shd_growy_append_formatted(g, "\tcase %s_TAG: {\n", name);
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
                        shd_growy_append_formatted(g, "\t\tnode->payload.%s.%s = string(arena, node->payload.%s.%s);\n", snake_name, op_name, snake_name, op_name);
                    else
                        shd_growy_append_formatted(g, "\t\tnode->payload.%s.%s = _shd_import_strings(arena, node->payload.%s.%s);\n", snake_name, op_name, snake_name, op_name);
                } else {
                    String cap = capitalize(class);
                    shd_growy_append_formatted(g, "\t\t{\n");
                    String extra = "";
                    if (list) {
                        shd_growy_append_formatted(g, "\t\t\tsize_t ops_count = node->payload.%s.%s.count;\n", snake_name, op_name);
                        shd_growy_append_formatted(g, "\t\t\tLARRAY(const Node*, ops, ops_count);\n");
                        shd_growy_append_formatted(g, "\t\t\tif (ops_count > 0) memcpy(ops, node->payload.%s.%s.nodes, sizeof(const Node*) * ops_count);\n", snake_name, op_name);
                        shd_growy_append_formatted(g, "\t\t\tfor (size_t i = 0; i < ops_count; i++) {\n");
                        shd_growy_append_formatted(g, "\t\t\tconst Node** pop = &ops[i];\n");
                        extra = "\t";
                    }
                    if (!list)
                        shd_growy_append_formatted(g, "\t\t\tconst Node** pop = &node->payload.%s.%s;\n", snake_name, op_name);

                    shd_growy_append_formatted(g, "\t\t\t*pop = _shd_fold_node_operand(%s_TAG, Nc%s, \"%s\", *pop);\n", name, cap, op_name);

                    if (!(json_object_get_boolean(json_object_object_get(op, "nullable")) || json_object_get_boolean(json_object_object_get(op, "ignore")))) {
                        shd_growy_append_formatted(g, "%s\t\t\tif (!*pop) {\n", extra);
                        shd_growy_append_formatted(g, "%s\t\t\t\tshd_error(\"operand '%s' of node '%s' cannot be null\");\n", extra, op_name, name);
                        shd_growy_append_formatted(g, "%s\t\t\t}\n", extra);
                    }

                    shd_growy_append_formatted(g, "%s\t\t\tif (arena->config.check_op_classes && *pop != NULL && !is_%s(*pop)) {\n", extra, class);
                    shd_growy_append_formatted(g, "%s\t\t\t\tshd_error_print(\"Invalid '%s' operand for node '%s', expected a %s\");\n", extra, op_name, name, class);
                    shd_growy_append_formatted(g, "%s\t\t\t\tshd_error_die();\n", extra);
                    shd_growy_append_formatted(g, "%s\t\t\t}\n", extra);
                    if (list) {
                        shd_growy_append_formatted(g, "\t\t\t}\n");
                        shd_growy_append_formatted(g, "\t\t\tnode->payload.%s.%s = shd_nodes(arena, ops_count, ops);\n", snake_name, op_name);
                    }
                    free((void*) cap);
                    shd_growy_append_formatted(g, "\t\t}\n");
                }
            }
        }
        shd_growy_append_formatted(g, "\t\tbreak;\n");
        shd_growy_append_formatted(g, "\t}\n", name);
        if (alloc)
            free(alloc);
    }
    shd_growy_append_formatted(g, "\t\tdefault: break;\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "}\n\n");
}

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    json_object* nodes = json_object_object_get(src, "nodes");
    generate_pre_construction_validation(g, src);
}
