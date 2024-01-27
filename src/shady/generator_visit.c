#include "generator.h"

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    json_object* nodes = json_object_object_get(src, "nodes");
    growy_append_formatted(g, "void visit_node_operands(Visitor* visitor, NodeClass exclude, const Node* node) {\n");
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
            growy_append_formatted(g, "\t\t%s payload = node->payload.%s;\n", name, snake_name);
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                String class = json_object_get_string(json_object_object_get(op, "class"));
                if (!class || strcmp(class, "string") == 0)
                    continue; // skip 'string' class and POD operands
                String class_cap = capitalize(class);
                bool list = json_object_get_boolean(json_object_object_get(op, "list"));
                bool ignore = json_object_get_boolean(json_object_object_get(op, "ignore"));
                if (!ignore) {
                    growy_append_formatted(g, "\t\tif ((exclude & Nc%s) == 0)\n", class_cap);
                    if (list)
                        growy_append_formatted(g, "\t\t\tvisit_ops(visitor, Nc%s, \"%s\", payload.%s);\n", class_cap, op_name, op_name);
                    else
                        growy_append_formatted(g, "\t\t\tvisit_op(visitor, Nc%s, \"%s\", payload.%s);\n", class_cap, op_name, op_name);
                }
                free(class_cap);
            }
        }
        growy_append_formatted(g, "\t\tbreak;\n");
        growy_append_formatted(g, "\t}\n", name);
        if (alloc)
            free(alloc);
    }
    growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "}\n\n");
}
