#include "generator.h"

static json_object* lookup_node_class(Data data, String name) {
    json_object* node_classes = json_object_object_get(data.shd, "node-classes");
    for (size_t i = 0; i < json_object_array_length(node_classes); i++) {
        json_object* class = json_object_array_get_idx(node_classes, i);
        String class_name = json_object_get_string(json_object_object_get(class, "name"));
        assert(class_name);
        if (strcmp(name, class_name) == 0)
            return class;
    }
    return NULL;
}

static String class_to_type(Data data, String class, bool list) {
    assert(class);
    if (strcmp(class, "string") == 0) {
        if (list)
            return "Strings";
        else
            return "String";
    }
    // check the class is valid
    if (!lookup_node_class(data, class)) {
        error_print("invalid node class '%s'\n", class);
        error_die();
    }
    return list ? "Nodes" : "const Node*";
}

static String get_type_for_operand(Data data, json_object* op) {
    String op_type = json_object_get_string(json_object_object_get(op, "type"));
    bool list = json_object_get_boolean(json_object_object_get(op, "list"));
    String op_class = NULL;
    if (!op_type) {
        op_class = json_object_get_string(json_object_object_get(op, "class"));
        op_type = class_to_type(data, op_class, list);
    }
    assert(op_type);
    return op_type;
}

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

static void generate_node_classes(Growy* g, json_object* classes) {
    assert(json_object_get_type(classes) == json_type_array);
    growy_append_formatted(g, "typedef enum {\n");
    for (size_t i = 0; i < json_object_array_length(classes); i++) {
        json_object* node_class = json_object_array_get_idx(classes, i);
        String name = json_object_get_string(json_object_object_get(node_class, "name"));
        String capitalized = capitalize(name);
        growy_append_formatted(g, "\tNc%s = 0b1", capitalized);
        for (int c = 0; c < i; c++)
            growy_append_string_literal(g, "0");
        growy_append_formatted(g, ",\n");
        free(capitalized);
    }
    growy_append_formatted(g, "} NodeClass;\n\n");
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

static void generate_node_payloads(Growy* g, Data data, json_object* nodes) {
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
                growy_append_formatted(g, "\t%s %s;\n", get_type_for_operand(data, op), op_name);
            }
            growy_append_formatted(g, "} %s;\n\n", name);
        }
    }
}

static void generate_node_type(Growy* g, json_object* nodes) {
    growy_append_formatted(g, "struct Node_ {\n");
    growy_append_formatted(g, "\tIrArena* arena;\n");
    growy_append_formatted(g, "\tconst Type* type;\n");
    growy_append_formatted(g, "\tNodeTag tag;\n");
    growy_append_formatted(g, "\tunion NodesUnion {\n");

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }

        json_object* ops = json_object_object_get(node, "ops");
        if (ops)
            growy_append_formatted(g, "\t\t%s %s;\n", name, snake_name);

        if (alloc)
            free(alloc);
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

void generate(Growy* g, Data data) {
    generate_header(g, data);

    generate_address_spaces(g, json_object_object_get(data.shd, "address-spaces"));

    json_object* node_classes = json_object_object_get(data.shd, "node-classes");
    generate_node_classes(g, node_classes);

    json_object* nodes = json_object_object_get(data.shd, "nodes");
    generate_node_tags(g, nodes);
    growy_append_formatted(g, "NodeClass get_node_class_from_tag(NodeTag tag);\n\n");
    generate_node_payloads(g, data, nodes);
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
    }
}
