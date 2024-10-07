#include "generator.h"

#include "generator.h"

static void generate_node_names_string_array(Growy* g, json_object* nodes) {
    shd_growy_append_formatted(g, "const char* node_tags[] = {\n");
    shd_growy_append_formatted(g, "\t\"invalid\",\n");
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            snake_name = to_snake_case(name);
            alloc = (void*) snake_name;
        }
        assert(name);
        shd_growy_append_formatted(g, "\t\"%s\",\n", snake_name);
        if (alloc)
            free(alloc);
    }
    shd_growy_append_formatted(g, "};\n\n");
}

static void generate_node_has_payload_array(Growy* g, json_object* nodes) {
    shd_growy_append_formatted(g, "const bool node_type_has_payload[]  = {\n");
    shd_growy_append_formatted(g, "\tfalse,\n");
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        json_object* ops = json_object_object_get(node, "ops");
        shd_growy_append_formatted(g, "\t%s,\n", ops ? "true" : "false");
    }
    shd_growy_append_formatted(g, "};\n\n");
}

static void generate_node_payload_hash_fn(Growy* g, json_object* src, json_object* nodes) {
    shd_growy_append_formatted(g, "KeyHash _shd_hash_node_payload(const Node* node) {\n");
    shd_growy_append_formatted(g, "\tKeyHash hash = 0;\n");
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
        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            shd_growy_append_formatted(g, "\tcase %s_TAG: {\n", name);
            shd_growy_append_formatted(g, "\t\t%s payload = node->payload.%s;\n", name, snake_name);
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                bool ignore = json_object_get_boolean(json_object_object_get(op, "ignore"));
                if (!ignore) {
                    shd_growy_append_formatted(g, "\t\thash = hash ^ shd_hash_murmur(&payload.%s, sizeof(payload.%s));\n", op_name, op_name);
                }
            }
            shd_growy_append_formatted(g, "\t\tbreak;\n");
            shd_growy_append_formatted(g, "\t}\n", name);
        }
        if (alloc)
            free(alloc);
    }
    shd_growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "\treturn hash;\n");
    shd_growy_append_formatted(g, "}\n");
}

static void generate_node_payload_cmp_fn(Growy* g, json_object* src, json_object* nodes) {
    shd_growy_append_formatted(g, "bool _shd_compare_node_payload(const Node* a, const Node* b) {\n");
    shd_growy_append_formatted(g, "\tbool eq = true;\n");
    shd_growy_append_formatted(g, "\tswitch (a->tag) { \n");
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
        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            shd_growy_append_formatted(g, "\tcase %s_TAG: {\n", name);
            shd_growy_append_formatted(g, "\t\t%s payload_a = a->payload.%s;\n", name, snake_name);
            shd_growy_append_formatted(g, "\t\t%s payload_b = b->payload.%s;\n", name, snake_name);
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                bool ignore = json_object_get_boolean(json_object_object_get(op, "ignore"));
                if (!ignore) {
                    shd_growy_append_formatted(g, "\t\teq &= memcmp(&payload_a.%s, &payload_b.%s, sizeof(payload_a.%s)) == 0;\n", op_name, op_name, op_name);
                }
            }
            shd_growy_append_formatted(g, "\t\tbreak;\n");
            shd_growy_append_formatted(g, "\t}\n", name);
        }
        if (alloc)
            free(alloc);
    }
    shd_growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "\treturn eq;\n");
    shd_growy_append_formatted(g, "}\n");
}

static void generate_node_is_nominal(Growy* g, json_object* nodes) {
    shd_growy_append_formatted(g, "bool shd_is_node_nominal(const Node* node) {\n");
    shd_growy_append_formatted(g, "\tswitch (node->tag) { \n");
    assert(json_object_get_type(nodes) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        if (json_object_get_boolean(json_object_object_get(node, "nominal"))) {
            shd_growy_append_formatted(g, "\t\tcase %s_TAG: return true;\n", name);
        }
    }
    shd_growy_append_formatted(g, "\t\tdefault: return false;\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "}\n");
}

static void generate_isa_for_class(Growy* g, json_object* nodes, String class, String capitalized_class, bool use_enum) {
    assert(json_object_get_type(nodes) == json_type_array);
    if (use_enum)
        shd_growy_append_formatted(g, "%sTag is_%s(const Node* node) {\n", capitalized_class, class);
    else
        shd_growy_append_formatted(g, "bool is_%s(const Node* node) {\n", class);
    shd_growy_append_formatted(g, "\tif (get_node_class_from_tag(node->tag) & Nc%s)\n", capitalized_class);
    if (use_enum) {
        shd_growy_append_formatted(g, "\t\treturn (%sTag) node->tag;\n", capitalized_class);
        shd_growy_append_formatted(g, "\treturn (%sTag) 0;\n", capitalized_class);
    } else {
        shd_growy_append_formatted(g, "\t\treturn true;\n", capitalized_class);
        shd_growy_append_formatted(g, "\treturn false;\n", capitalized_class);
    }
    shd_growy_append_formatted(g, "}\n\n");
}

static bool is_of(json_object* node, String class_name) {
    switch (json_object_get_type(node)) {
        case json_type_array: {
            for (size_t i = 0; i < json_object_array_length(node); i++)
                if (is_of(json_object_array_get_idx(node, i), class_name))
                    return true;
            break;
        }
        case json_type_string: return strcmp(json_object_get_string(node), class_name) == 0;
        default: break;
    }
    return false;
}

static void generate_getters_for_class(Growy* g, json_object* src, json_object* nodes, json_object* node_class) {
    String class_name = json_object_get_string(json_object_object_get(node_class, "name"));
    json_object* class_ops = json_object_object_get(node_class, "ops");
    if (!class_ops)
        return;
    assert(json_object_get_type(class_ops) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(class_ops); i++) {
        json_object* operand = json_object_array_get_idx(class_ops, i);
        String operand_name = json_object_get_string(json_object_object_get(operand, "name"));
        assert(operand_name);
        shd_growy_append_formatted(g, "%s get_%s_%s(const Node* node) {\n", get_type_for_operand(src, operand), class_name, operand_name);
        shd_growy_append_formatted(g, "\tswitch(node->tag) {\n");
        for (size_t j = 0; j < json_object_array_length(nodes); j++) {
            json_object* node = json_object_array_get_idx(nodes, j);
            if (is_of(json_object_object_get(node, "class"), class_name)) {
                String node_name = json_object_get_string(json_object_object_get(node, "name"));
                shd_growy_append_formatted(g, "\t\tcase %s_TAG: ", node_name);
                String node_snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
                assert(node_snake_name);
                shd_growy_append_formatted(g, "return node->payload.%s.%s;\n", node_snake_name, operand_name);
            }
        }
        shd_growy_append_formatted(g, "\t\tdefault: break;\n");
        shd_growy_append_formatted(g, "\t}\n");
        shd_growy_append_formatted(g, "\tassert(false);\n");
        shd_growy_append_formatted(g, "}\n\n");
    }
}

void generate_address_space_name_fn(Growy* g, json_object* address_spaces) {
    shd_growy_append_formatted(g, "String get_address_space_name(AddressSpace as) {\n");
    shd_growy_append_formatted(g, "\tswitch (as) {\n");
    for (size_t i = 0; i < json_object_array_length(address_spaces); i++) {
        json_object* as = json_object_array_get_idx(address_spaces, i);
        String name = json_object_get_string(json_object_object_get(as, "name"));
        shd_growy_append_formatted(g, "\t\t case As%s: return \"%s\";\n", name, name);
    }
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "}\n");
}

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    json_object* nodes = json_object_object_get(src, "nodes");
    generate_address_space_name_fn(g, json_object_object_get(src, "address-spaces"));
    generate_node_names_string_array(g, nodes);
    generate_node_is_nominal(g, nodes);
    generate_node_has_payload_array(g, nodes);
    generate_node_payload_hash_fn(g, src, nodes);
    generate_node_payload_cmp_fn(g, src, nodes);
    generate_bit_enum_classifier(g, "get_node_class_from_tag", "NodeClass", "Nc", "NodeTag", "", "_TAG", nodes);

    json_object* node_classes = json_object_object_get(src, "node-classes");
    for (size_t i = 0; i < json_object_array_length(node_classes); i++) {
        json_object* node_class = json_object_array_get_idx(node_classes, i);
        String name = json_object_get_string(json_object_object_get(node_class, "name"));
        assert(name);

        generate_getters_for_class(g, src, nodes, node_class);

        json_object* generate_enum = json_object_object_get(node_class, "generate-enum");
        String capitalized = capitalize(name);
        generate_isa_for_class(g, nodes, name, capitalized, !generate_enum || json_object_get_boolean(generate_enum));
        free((void*) capitalized);
    }
}
