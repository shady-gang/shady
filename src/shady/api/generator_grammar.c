#include "generator.h"

static void generate_address_spaces(Growy* g, json_object* address_spaces) {
    growy_append_formatted(g, "typedef enum AddressSpace_ {\n");
    for (size_t i = 0; i < json_object_array_length(address_spaces); i++) {
        json_object* as = json_object_array_get_idx(address_spaces, i);
        String name = json_object_get_string(json_object_object_get(as, "name"));
        add_comments(g, "\t", json_object_object_get(as, "description"));
        growy_append_formatted(g, "\tAs%s,\n", name);
    }
    growy_append_formatted(g, "\tNumAddressSpaces,\n");
    growy_append_formatted(g, "} AddressSpace;\n\n");

    growy_append_formatted(g, "static inline bool is_physical_as(AddressSpace as) {\n");
    growy_append_formatted(g, "\tswitch(as) {\n");
    for (size_t i = 0; i < json_object_array_length(address_spaces); i++) {
        json_object* as = json_object_array_get_idx(address_spaces, i);
        String name = json_object_get_string(json_object_object_get(as, "name"));
        if (json_object_get_boolean(json_object_object_get(as, "physical")))
            growy_append_formatted(g, "\t\tcase As%s: return true;\n", name);
    }
    growy_append_formatted(g, "\t\tdefault: return false;\n");
    growy_append_formatted(g, "\t}\n");
    growy_append_formatted(g, "}\n\n");
}

static void generate_node_tags(Growy* g, json_object* nodes) {
    assert(json_object_get_type(nodes) == json_type_array);
    growy_append_formatted(g, "typedef enum {\n");
    growy_append_formatted(g, "\tInvalidNode_TAG,\n");

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);
        json_object* ops = json_object_object_get(node, "ops");
        if (!ops)
            add_comments(g, "\t", json_object_object_get(node, "description"));

        growy_append_formatted(g, "\t%s_TAG,\n", name);
    }
    growy_append_formatted(g, "} NodeTag;\n\n");
}

static void generate_node_payloads(Growy* g, json_object* src, json_object* nodes) {
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        json_object* ops = json_object_object_get(node, "ops");
        if (ops) {
            assert(json_object_get_type(ops) == json_type_array);
            add_comments(g, "", json_object_object_get(node, "description"));
            growy_append_formatted(g, "typedef struct {\n");
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                growy_append_formatted(g, "\t%s %s;\n", get_type_for_operand(src, op), op_name);
            }
            growy_append_formatted(g, "} %s;\n\n", name);
        }
    }
}

static void generate_node_type(Growy* g, json_object* nodes) {
    growy_append_formatted(g, "struct Node_ {\n");
    growy_append_formatted(g, "\tIrArena* arena;\n");
    growy_append_formatted(g, "\tNodeId id;\n");
    growy_append_formatted(g, "\tconst Type* type;\n");
    growy_append_formatted(g, "\tNodeTag tag;\n");
    growy_append_formatted(g, "\tunion NodesUnion {\n");

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        assert(snake_name);

        json_object* ops = json_object_object_get(node, "ops");
        if (ops)
            growy_append_formatted(g, "\t\t%s %s;\n", name, snake_name);
    }

    growy_append_formatted(g, "\t} payload;\n");
    growy_append_formatted(g, "};\n\n");
}

static void generate_node_tags_for_class(Growy* g, json_object* nodes, String class, String capitalized_class) {
    assert(json_object_get_type(nodes) == json_type_array);
    growy_append_formatted(g, "typedef enum {\n");
    if (starts_with_vowel(class))
        growy_append_formatted(g, "\tNotAn%s = 0,\n", capitalized_class);
    else
        growy_append_formatted(g, "\tNotA%s = 0,\n", capitalized_class);

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);
        json_object* nclass = json_object_object_get(node, "class");
        switch (json_object_get_type(nclass)) {
            case json_type_null:
                break;
            case json_type_string:
                if (nclass && strcmp(json_object_get_string(nclass), class) == 0)
                    growy_append_formatted(g, "\t%s_%s_TAG = %s_TAG,\n", capitalized_class, name, name);
                break;
            case json_type_array: {
                for (size_t j = 0; j < json_object_array_length(nclass); j++) {
                    if (nclass && strcmp(json_object_get_string(json_object_array_get_idx(nclass, j)), class) == 0) {
                        growy_append_formatted(g, "\t%s_%s_TAG = %s_TAG,\n", capitalized_class, name, name);
                        break;
                    }
                }
                break;
            }
            case json_type_boolean:
            case json_type_double:
            case json_type_int:
            case json_type_object:
                error_print("Invalid datatype for a node's 'class' attribute");
        }

    }
    growy_append_formatted(g, "} %sTag;\n\n", capitalized_class);
}

static void generate_header_getters_for_class(Growy* g, json_object* src, json_object* node_class) {
    String class_name = json_object_get_string(json_object_object_get(node_class, "name"));
    json_object* class_ops = json_object_object_get(node_class, "ops");
    if (!class_ops)
        return;
    assert(json_object_get_type(class_ops) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(class_ops); i++) {
        json_object* operand = json_object_array_get_idx(class_ops, i);
        String operand_name = json_object_get_string(json_object_object_get(operand, "name"));
        assert(operand_name);
        growy_append_formatted(g, "%s get_%s_%s(const Node* node);\n", get_type_for_operand(src, operand), class_name, operand_name);
    }
}

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    generate_address_spaces(g, json_object_object_get(src, "address-spaces"));

    json_object* node_classes = json_object_object_get(src, "node-classes");
    generate_bit_enum(g, "NodeClass", "Nc", node_classes);

    json_object* nodes = json_object_object_get(src, "nodes");
    generate_node_tags(g, nodes);
    growy_append_formatted(g, "NodeClass get_node_class_from_tag(NodeTag tag);\n\n");
    generate_node_payloads(g, src, nodes);
    generate_node_type(g, nodes);
    generate_node_ctor(g, nodes, false);

    for (size_t i = 0; i < json_object_array_length(node_classes); i++) {
        json_object* node_class = json_object_array_get_idx(node_classes, i);
        String name = json_object_get_string(json_object_object_get(node_class, "name"));
        assert(name);
        json_object* generate_enum = json_object_object_get(node_class, "generate-enum");
        if (!generate_enum || json_object_get_boolean(generate_enum)) {
            String capitalized = capitalize(name);
            generate_node_tags_for_class(g, nodes, name, capitalized);
            growy_append_formatted(g, "%sTag is_%s(const Node*);\n", capitalized, name);
            free(capitalized);
        } else {
            growy_append_formatted(g, "bool is_%s(const Node*);\n", name);
        }

        generate_header_getters_for_class(g, src, node_class);
    }
}
