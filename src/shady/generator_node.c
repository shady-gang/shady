#include "generator.h"

#include "generator.h"

static void generate_node_names_string_array(Growy* g, json_object* nodes) {
    growy_append_formatted(g, "const char* node_tags[] = {\n");
    growy_append_formatted(g, "\t\"invalid\",\n");
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }
        assert(name);
        growy_append_formatted(g, "\t\"%s\",\n", snake_name);
        if (alloc)
            free(alloc);
    }
    growy_append_formatted(g, "};\n\n");
}

static void generate_node_has_payload_array(Growy* g, json_object* nodes) {
    growy_append_formatted(g, "const bool node_type_has_payload[]  = {\n");
    growy_append_formatted(g, "\tfalse,\n");
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        json_object* ops = json_object_object_get(node, "ops");
        growy_append_formatted(g, "\t%s,\n", ops ? "true" : "false");
    }
    growy_append_formatted(g, "};\n\n");
}

static void generate_node_payload_hash_fn(Growy* g, Data data, json_object* nodes) {
    growy_append_formatted(g, "KeyHash hash_node_payload(const Node* node) {\n");
    growy_append_formatted(g, "\tKeyHash hash = 0;\n");
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
        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            growy_append_formatted(g, "\tcase %s_TAG: {\n", name);
            growy_append_formatted(g, "\t\t%s payload = node->payload.%s;\n", name, snake_name);
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                bool ignore = json_object_get_boolean(json_object_object_get(op, "ignore"));
                if (!ignore) {
                    growy_append_formatted(g, "\t\thash = hash ^ hash_murmur(&payload.%s, sizeof(payload.%s));\n", op_name, op_name);
                }
            }
            growy_append_formatted(g, "\t\tbreak;\n");
            growy_append_formatted(g, "\t}\n", name);
        }
        if (alloc)
            free(alloc);
    }
    growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "\treturn hash;\n");
    growy_append_formatted(g, "}\n");
}

static void generate_node_payload_cmp_fn(Growy* g, Data data, json_object* nodes) {
    growy_append_formatted(g, "bool compare_node_payload(const Node* a, const Node* b) {\n");
    growy_append_formatted(g, "\tbool eq = true;\n");
    growy_append_formatted(g, "\tswitch (a->tag) { \n");
    assert(json_object_get_type(nodes) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }
        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            growy_append_formatted(g, "\tcase %s_TAG: {\n", name);
            growy_append_formatted(g, "\t\t%s payload_a = a->payload.%s;\n", name, snake_name);
            growy_append_formatted(g, "\t\t%s payload_b = b->payload.%s;\n", name, snake_name);
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                bool ignore = json_object_get_boolean(json_object_object_get(op, "ignore"));
                if (!ignore) {
                    growy_append_formatted(g, "\t\teq &= memcmp(&payload_a.%s, &payload_b.%s, sizeof(payload_a.%s)) == 0;\n", op_name, op_name, op_name);
                }
            }
            growy_append_formatted(g, "\t\tbreak;\n");
            growy_append_formatted(g, "\t}\n", name);
        }
        if (alloc)
            free(alloc);
    }
    growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "\treturn eq;\n");
    growy_append_formatted(g, "}\n");
}

static void generate_isa_for_class(Growy* g, json_object* nodes, String class, String capitalized_class, bool use_enum) {
    assert(json_object_get_type(nodes) == json_type_array);
    if (use_enum)
        growy_append_formatted(g, "%sTag is_%s(const Node* node) {\n", capitalized_class, class);
    else
        growy_append_formatted(g, "bool is_%s(const Node* node) {\n", class);
    growy_append_formatted(g, "\tif (get_node_class_from_tag(node->tag) & Nc%s)\n", capitalized_class);
    if (use_enum) {
        growy_append_formatted(g, "\t\treturn (%sTag) node->tag;\n", capitalized_class);
        growy_append_formatted(g, "\treturn (%sTag) 0;\n", capitalized_class);
    } else {
        growy_append_formatted(g, "\t\treturn true;\n", capitalized_class);
        growy_append_formatted(g, "\treturn false;\n", capitalized_class);
    }
    growy_append_formatted(g, "}\n\n");
}

static void generate_primops_names_array(Growy* g, json_object* primops) {
    growy_append_string(g, "const char* primop_names[] = {\n");

    for (size_t i = 0; i < json_object_array_length(primops); i++) {
        json_object* node = json_object_array_get_idx(primops, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        growy_append_formatted(g, "\"%s\",", name);
    }

    growy_append_string(g, "\n};\n");
}

static void generate_primops_side_effects_array(Growy* g, json_object* primops) {
    growy_append_string(g, "const bool primop_side_effects[] = {\n");

    for (size_t i = 0; i < json_object_array_length(primops); i++) {
        json_object* node = json_object_array_get_idx(primops, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        bool side_effects = json_object_get_boolean(json_object_object_get(node, "side-effects"));
        if (side_effects)
            growy_append_string(g, "true, ");
        else
            growy_append_string(g, "false, ");
    }

    growy_append_string(g, "\n};\n");
}

void generate_address_space_name_fn(Growy* g, json_object* address_spaces) {
    growy_append_formatted(g, "String get_address_space_name(AddressSpace as) {\n");
    growy_append_formatted(g, "\tswitch (as) {\n");
    for (size_t i = 0; i < json_object_array_length(address_spaces); i++) {
        json_object* as = json_object_array_get_idx(address_spaces, i);
        String name = json_object_get_string(json_object_object_get(as, "name"));
        growy_append_formatted(g, "\t\t case As%s: return \"%s\";\n", name, name);
    }
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "}\n");
}

void generate(Growy* g, Data data) {
    generate_header(g, data);

    json_object* nodes = json_object_object_get(data.shd, "nodes");
    generate_address_space_name_fn(g, json_object_object_get(data.shd, "address-spaces"));
    generate_node_names_string_array(g, nodes);
    generate_node_has_payload_array(g, nodes);
    generate_node_payload_hash_fn(g, data, nodes);
    generate_node_payload_cmp_fn(g, data, nodes);
    generate_bit_enum_classifier(g, "get_node_class_from_tag", "NodeClass", "Nc", "NodeTag", "", "_TAG", nodes);

    json_object* node_classes = json_object_object_get(data.shd, "node-classes");
    for (size_t i = 0; i < json_object_array_length(node_classes); i++) {
        json_object* node_class = json_object_array_get_idx(node_classes, i);
        String name = json_object_get_string(json_object_object_get(node_class, "name"));
        assert(name);

        json_object* generate_enum = json_object_object_get(node_class, "generate-enum");
        String capitalized = capitalize(name);
        generate_isa_for_class(g, nodes, name, capitalized, !generate_enum || json_object_get_boolean(generate_enum));
        free(capitalized);
    }

    json_object* primops = json_object_object_get(data.shd, "prim-ops");
    generate_primops_names_array(g, primops);
    generate_primops_side_effects_array(g, primops);
}
