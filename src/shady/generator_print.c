#include "generator.h"

void generate_node_print_fns(Growy* g, json_object* src) {
    json_object* nodes = json_object_object_get(src, "nodes");
    shd_growy_append_formatted(g, "void _shd_print_node_generated(PrinterCtx* ctx, const Node* node) {\n");
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
        shd_growy_append_formatted(g, "\t\tshd_print(ctx->printer, GREEN);\n");
        shd_growy_append_formatted(g, "\t\tshd_print(ctx->printer, \"%s\");\n", name);
        shd_growy_append_formatted(g, "\t\tshd_print(ctx->printer, RESET);\n");
        shd_growy_append_formatted(g, "\t\tshd_print(ctx->printer, \"(\");\n");
        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            bool first = true;
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                bool ignore = json_object_get_boolean(json_object_object_get(op, "ignore"));
                bool tail = json_object_get_boolean(json_object_object_get(op, "tail"));
                if (ignore)
                    continue;

                if (first) {
                    first = false;
                } else {
                    shd_growy_append_formatted(g, "\t\tshd_print(ctx->printer, \", \");\n");
                }

                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                String op_class = json_object_get_string(json_object_object_get(op, "class"));
                if (op_class && strcmp(op_class, "string") != 0) {
                    bool is_list = json_object_get_boolean(json_object_object_get(op, "list"));
                    String cap_class = capitalize(op_class);
                    if (is_list) {
                        shd_growy_append_formatted(g, "\t\t{\n");
                        shd_growy_append_formatted(g, "\t\t\t_shd_print_node_operand_list(ctx, node, \"%s\", Nc%s, node->payload.%s.%s);\n", op_name, cap_class, snake_name, op_name);
                        // growy_append_formatted(g, "\t\t\tsize_t count = node->payload.%s.%s.count;\n", snake_name, op_name);
                        // growy_append_formatted(g, "\t\t\tfor (size_t i = 0; i < count; i++) {\n");
                        // growy_append_formatted(g, "\t\t\t\tprint_node_operand(printer, node, \"%s\", Nc%s, i, node->payload.%s.%s.nodes[i], config);\n", op_name, cap_class, snake_name, op_name);
                        // growy_append_formatted(g, "\t\t\t}\n");
                        shd_growy_append_formatted(g, "\t\t}\n");
                    } else {
                        shd_growy_append_formatted(g, "\t\t{\n");
                        shd_growy_append_formatted(g, "\t\t\t_shd_print_node_operand(ctx, node, \"%s\", Nc%s, node->payload.%s.%s);\n", op_name, cap_class, snake_name, op_name);
                        shd_growy_append_formatted(g, "\t\t}\n");
                    }
                    free((void*) cap_class);
                } else {
                    String op_type = json_object_get_string(json_object_object_get(op, "type"));
                    if (!op_type) {
                        assert(op_class && strcmp(op_class, "string") == 0);
                        if (json_object_get_boolean(json_object_object_get(op, "list")))
                            op_type = "Strings";
                        else
                            op_type = "String";
                    }
                    char* s = strdup(op_type);
                    for (size_t k = 0; k < strlen(op_type); k++) {
                        if (!isalnum(s[k]))
                            s[k] = '_';
                    }
                    shd_growy_append_formatted(g, "\t\t_shd_print_node_operand_%s(ctx, node, \"%s\", node->payload.%s.%s);\n", s, op_name, snake_name, op_name);
                    free(s);
                }
            }
        }
        shd_growy_append_formatted(g, "\t\tshd_print(ctx->printer, \")\");\n");
        shd_growy_append_formatted(g, "\t\tbreak;\n");
        shd_growy_append_formatted(g, "\t}\n", name);
        if (alloc)
            free(alloc);
    }
    shd_growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "}\n");
}

void generate(Growy* g, json_object* src) {
    generate_header(g, src);
    generate_node_print_fns(g, src);
}
