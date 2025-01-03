#include "generator.h"

static void generate_address_spaces(Growy* g, json_object* address_spaces) {
    shd_growy_append_formatted(g, "typedef enum AddressSpace_ {\n");
    for (size_t i = 0; i < json_object_array_length(address_spaces); i++) {
        json_object* as = json_object_array_get_idx(address_spaces, i);
        String name = json_object_get_string(json_object_object_get(as, "name"));
        add_comments(g, "\t", json_object_object_get(as, "description"));
        shd_growy_append_formatted(g, "\tAs%s,\n", name);
    }
    shd_growy_append_formatted(g, "\tNumAddressSpaces,\n");
    shd_growy_append_formatted(g, "} AddressSpace;\n\n");
}

static void generate_node_tags(Growy* g, json_object* nodes) {
    assert(json_object_get_type(nodes) == json_type_array);
    shd_growy_append_formatted(g, "typedef enum {\n");
    shd_growy_append_formatted(g, "\tInvalidNode_TAG,\n");

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);
        json_object* ops = json_object_object_get(node, "ops");
        if (!ops)
            add_comments(g, "\t", json_object_object_get(node, "description"));

        shd_growy_append_formatted(g, "\t%s_TAG,\n", name);
    }
    shd_growy_append_formatted(g, "} NodeTag;\n\n");
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
            shd_growy_append_formatted(g, "typedef struct SHADY_DESIGNATED_INIT {\n");
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                shd_growy_append_formatted(g, "\t%s %s;\n", get_type_for_operand(src, op), op_name);
            }
            shd_growy_append_formatted(g, "} %s;\n\n", name);
        }
    }
}

static void generate_node_type(Growy* g, json_object* nodes) {
    shd_growy_append_formatted(g, "struct Node_ {\n");
    shd_growy_append_formatted(g, "\tIrArena* arena;\n");
    shd_growy_append_formatted(g, "\tNodeId id;\n");
    shd_growy_append_formatted(g, "\tconst Type* type;\n");
    shd_growy_append_formatted(g, "\tNodeTag tag;\n");
    shd_growy_append_formatted(g, "\tunion NodesUnion {\n");

    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        assert(snake_name);

        json_object* ops = json_object_object_get(node, "ops");
        if (ops)
            shd_growy_append_formatted(g, "\t\t%s %s;\n", name, snake_name);
    }

    shd_growy_append_formatted(g, "\t} payload;\n");
    shd_growy_append_formatted(g, "};\n\n");
}

static void generate_node_tags_for_class(Growy* g, json_object* nodes, String class, String capitalized_class) {
    assert(json_object_get_type(nodes) == json_type_array);
    shd_growy_append_formatted(g, "typedef enum {\n");
    if (starts_with_vowel(class))
        shd_growy_append_formatted(g, "\tNotAn%s = 0,\n", capitalized_class);
    else
        shd_growy_append_formatted(g, "\tNotA%s = 0,\n", capitalized_class);

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
                    shd_growy_append_formatted(g, "\t%s_%s_TAG = %s_TAG,\n", capitalized_class, name, name);
                break;
            case json_type_array: {
                for (size_t j = 0; j < json_object_array_length(nclass); j++) {
                    if (nclass && strcmp(json_object_get_string(json_object_array_get_idx(nclass, j)), class) == 0) {
                        shd_growy_append_formatted(g, "\t%s_%s_TAG = %s_TAG,\n", capitalized_class, name, name);
                        break;
                    }
                }
                break;
            }
            case json_type_boolean:
            case json_type_double:
            case json_type_int:
            case json_type_object:
                shd_error_print("Invalid datatype for a node's 'class' attribute");
        }

    }
    shd_growy_append_formatted(g, "} %sTag;\n\n", capitalized_class);
}

static void generate_isa_for_class(Growy* g, json_object* nodes, String class, String capitalized_class, bool use_enum) {
    assert(json_object_get_type(nodes) == json_type_array);
    if (use_enum)
        shd_growy_append_formatted(g, "static inline %sTag is_%s(const Node* node) {\n", capitalized_class, class);
    else
        shd_growy_append_formatted(g, "static inline bool is_%s(const Node* node) {\n", class);
    shd_growy_append_formatted(g, "\tif (shd_get_node_class_from_tag(node->tag) & Nc%s)\n", capitalized_class);
    if (use_enum) {
        shd_growy_append_formatted(g, "\t\treturn (%sTag) node->tag;\n", capitalized_class);
        shd_growy_append_formatted(g, "\treturn (%sTag) 0;\n", capitalized_class);
    } else {
        shd_growy_append_formatted(g, "\t\treturn true;\n", capitalized_class);
        shd_growy_append_formatted(g, "\treturn false;\n", capitalized_class);
    }
    shd_growy_append_formatted(g, "}\n\n");
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
        shd_growy_append_formatted(g, "%s get_%s_%s(const Node* node);\n", get_type_for_operand(src, operand), class_name, operand_name);
    }
}

void generate_node_ctor(Growy* g, json_object* src, json_object* nodes) {
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        if (has_custom_ctor(node))
            continue;

        if (i > 0)
            shd_growy_append_formatted(g, "\n");

        String snake_name = json_object_get_string(json_object_object_get(node, "snake_name"));
        const void* alloc = NULL;
        if (!snake_name) {
            alloc = snake_name = to_snake_case(name);
        }

        json_object* ops = json_object_object_get(node, "ops");
        if (ops)
            shd_growy_append_formatted(g, "static inline const Node* %s(IrArena* arena, %s payload)", snake_name, name);
        else
            shd_growy_append_formatted(g, "static inline const Node* %s(IrArena* arena)", snake_name);

        shd_growy_append_formatted(g, " {\n");
        shd_growy_append_formatted(g, "\tNode node;\n");
        shd_growy_append_formatted(g, "\tmemset((void*) &node, 0, sizeof(Node));\n");
        shd_growy_append_formatted(g, "\tnode = (Node) {\n");
        shd_growy_append_formatted(g, "\t\t.arena = arena,\n");
        shd_growy_append_formatted(g, "\t\t.type = NULL,\n");
        shd_growy_append_formatted(g, "\t\t.tag = %s_TAG,\n", name);
        if (ops)
            shd_growy_append_formatted(g, "\t\t.payload = { .%s = payload },\n", snake_name);
        shd_growy_append_formatted(g, "\t};\n");
        shd_growy_append_formatted(g, "\treturn _shd_create_node_helper(arena, node, NULL);\n");
        shd_growy_append_formatted(g, "}\n");

        // Generate helper variant
        if (ops) {
            shd_growy_append_formatted(g, "static inline const Node* %s_helper(IrArena* arena, ", snake_name);
            bool first = true;
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                if (json_object_get_boolean(json_object_object_get(op, "ignore")))
                    continue;
                if (first)
                    first = false;
                else
                    shd_growy_append_formatted(g, ", ");
                shd_growy_append_formatted(g, "%s %s", get_type_for_operand(src, op), op_name);
            }
            shd_growy_append_formatted(g, ") {\n");
            shd_growy_append_formatted(g, "\treturn %s(arena, (%s) {", snake_name, name);
            first = true;
            for (size_t j = 0; j < json_object_array_length(ops); j++) {
                json_object* op = json_object_array_get_idx(ops, j);
                String op_name = json_object_get_string(json_object_object_get(op, "name"));
                if (json_object_get_boolean(json_object_object_get(op, "ignore")))
                    continue;
                if (first)
                    first = false;
                else
                    shd_growy_append_formatted(g, ", ");
                shd_growy_append_formatted(g, ".%s = %s", op_name, op_name);
            }
            shd_growy_append_formatted(g, "});\n");
            shd_growy_append_formatted(g, "}\n");
        }

        if (alloc)
            free((void*) alloc);
    }
    shd_growy_append_formatted(g, "\n");
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
        shd_growy_append_formatted(g, "static inline %s get_%s_%s(const Node* node) {\n", get_type_for_operand(src, operand), class_name, operand_name);
        shd_growy_append_formatted(g, "\tswitch(node->tag) {\n");
        for (size_t j = 0; j < json_object_array_length(nodes); j++) {
            json_object* node = json_object_array_get_idx(nodes, j);
            if (find_in_set(json_object_object_get(node, "class"), class_name)) {
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

void generate(Growy* g, json_object* src) {
    generate_header(g, src);

    generate_address_spaces(g, json_object_object_get(src, "address-spaces"));

    json_object* node_classes = json_object_object_get(src, "node-classes");
    generate_bit_enum(g, "NodeClass", "Nc", node_classes);

    json_object* nodes = json_object_object_get(src, "nodes");
    generate_node_tags(g, nodes);
    shd_growy_append_formatted(g, "NodeClass shd_get_node_class_from_tag(NodeTag tag);\n\n");
    generate_node_payloads(g, src, nodes);
    generate_node_type(g, nodes);

    shd_growy_append_formatted(g, "#include <string.h>\n");
    shd_growy_append_formatted(g, "#include <assert.h>\n");
    shd_growy_append_formatted(g, "Node* _shd_create_node_helper(IrArena* arena, Node node, bool* pfresh);\n");
    generate_node_ctor(g, src, nodes);

    for (size_t i = 0; i < json_object_array_length(node_classes); i++) {
        json_object* node_class = json_object_array_get_idx(node_classes, i);
        String name = json_object_get_string(json_object_object_get(node_class, "name"));
        assert(name);
        json_object* generate_enum = json_object_object_get(node_class, "generate-enum");
        String capitalized = capitalize(name);

        if (!generate_enum || json_object_get_boolean(generate_enum)) {
            generate_node_tags_for_class(g, nodes, name, capitalized);
        }

        //generate_header_getters_for_class(g, src, node_class);
        generate_getters_for_class(g, src, nodes, node_class);
        generate_isa_for_class(g, nodes, name, capitalized, !generate_enum || json_object_get_boolean(generate_enum));
        free((void*) capitalized);
    }
}
