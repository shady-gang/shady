#include "generator.h"

static void generate_can_be_default_rewritten_fn(Growy* g, json_object* nodes) {
    growy_append_formatted(g, "static bool can_be_default_rewritten(NodeTag tag) {\n");
    growy_append_formatted(g, "\tswitch (tag) { \n");
    assert(json_object_get_type(nodes) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        if (has_custom_ctor(node))
            continue;
        String name = json_object_get_string(json_object_object_get(node, "name"));
        growy_append_formatted(g, "\t\tcase %s_TAG: return true;\n", name);
    }
    growy_append_formatted(g, "\t\tdefault: return false;\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "}\n\n");
}

static void generate_rewriter_default_fns(Growy* g, json_object* nodes) {
    growy_append_formatted(g, "static const Node* recreate_node_identity_generated(Rewriter* rewriter, const Node* node) {\n");
    growy_append_formatted(g, "\tswitch (node->tag) { \n");
    assert(json_object_get_type(nodes) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        if (has_custom_ctor(node))
            continue;

        String name = json_object_get_string(json_object_object_get(node, "name"));
        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }
        growy_append_formatted(g, "\t\tcase %s_TAG: {\n", name);
        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            growy_append_formatted(g, "\t\t\t%s old_payload = node->payload.%s;\n", name, snake_name);
            growy_append_formatted(g, "\t\t\t%s payload = { 0 };\n", name, snake_name);
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                bool list = json_object_get_boolean(json_object_object_get(op, "list"));
                bool ignore = json_object_get_boolean(json_object_object_get(op, "ignore"));
                if (ignore)
                    continue;
                String class = json_object_get_string(json_object_object_get(op, "class"));
                if (!class) {
                    assert(!list);
                    growy_append_formatted(g, "\t\t\tpayload.%s = old_payload.%s;\n", op_name, op_name);
                    continue;
                }
                if (strcmp(class, "string") == 0) {
                    if (list)
                        growy_append_formatted(g, "\t\t\tpayload.%s = strings(rewriter->dst_arena, old_payload.%s.count, old_payload.%s.strings);\n", op_name, op_name, op_name);
                    else
                        growy_append_formatted(g, "\t\t\tpayload.%s = string(rewriter->dst_arena, old_payload.%s);\n", op_name, op_name);
                    continue;
                }

                String class_cap = capitalize(class);
                if (list)
                    growy_append_formatted(g, "\t\t\tpayload.%s = rewrite_ops_helper(rewriter, Nc%s, \"%s\", old_payload.%s);\n", op_name, class_cap, op_name, op_name);
                else
                    growy_append_formatted(g, "\t\t\tpayload.%s = rewrite_op_helper(rewriter, Nc%s, \"%s\", old_payload.%s);\n", op_name, class_cap, op_name, op_name);
                free(class_cap);
            }
            growy_append_formatted(g, "\t\t\treturn %s(rewriter->dst_arena, payload);\n", snake_name);
        } else
            growy_append_formatted(g, "\t\t\treturn %s(rewriter->dst_arena);\n", snake_name);
        growy_append_formatted(g, "\t\t}\n", name);
        if (alloc)
            free(alloc);
    }
    growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "}\n\n");
}

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    json_object* nodes = json_object_object_get(src, "nodes");
    generate_can_be_default_rewritten_fn(g, nodes);
    generate_rewriter_default_fns(g, nodes);
}
