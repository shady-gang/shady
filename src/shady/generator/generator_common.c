#include "generator.h"

bool has_custom_ctor(json_object* node) {
    String constructor = json_object_get_string(json_object_object_get(node, "constructor"));
    return (constructor && strcmp(constructor, "custom") == 0);
}

json_object* lookup_node_class(json_object* src, String name) {
    json_object* node_classes = json_object_object_get(src, "node-classes");
    for (size_t i = 0; i < json_object_array_length(node_classes); i++) {
        json_object* class = json_object_array_get_idx(node_classes, i);
        String class_name = json_object_get_string(json_object_object_get(class, "name"));
        assert(class_name);
        if (strcmp(name, class_name) == 0)
            return class;
    }
    return NULL;
}

String class_to_type(json_object* src, String class, bool list) {
    assert(class);
    if (strcmp(class, "string") == 0) {
        if (list)
            return "Strings";
        else
            return "String";
    }
    // check the class is valid
    if (!lookup_node_class(src, class)) {
        shd_error_print("invalid node class '%s'\n", class);
        shd_error_die();
    }
    return list ? "Nodes" : "const Node*";
}

bool find_in_set(json_object* node, String class_name) {
    switch (json_object_get_type(node)) {
        case json_type_array: {
            for (size_t i = 0; i < json_object_array_length(node); i++)
                if (find_in_set(json_object_array_get_idx(node, i), class_name))
                    return true;
            break;
        }
        case json_type_string: return strcmp(json_object_get_string(node), class_name) == 0;
        default: break;
    }
    return false;
}

String get_type_for_operand(json_object* src, json_object* op) {
    String op_type = json_object_get_string(json_object_object_get(op, "type"));
    bool list = json_object_get_boolean(json_object_object_get(op, "list"));
    String op_class = NULL;
    if (!op_type) {
        op_class = json_object_get_string(json_object_object_get(op, "class"));
        op_type = class_to_type(src, op_class, list);
    }
    assert(op_type);
    return op_type;
}

void preprocess(json_object* src) {
    json_object* nodes = json_object_object_get(src, "nodes");
    for (size_t i = 0; i < json_object_array_length(nodes); i++) {
        json_object* node = json_object_array_get_idx(nodes, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        json_object* snake_name = json_object_object_get(node, "snake_name");
        if (!snake_name) {
            String tmp = to_snake_case(name);
            json_object* generated_snake_name = json_object_new_string(tmp);
            json_object_object_add(node, "snake_name", generated_snake_name);
            free((void*) tmp);
        }
    }
}

void generate_bit_enum(Growy* g, String enum_type_name, String enum_case_prefix, json_object* cases) {
    assert(json_object_get_type(cases) == json_type_array);
    shd_growy_append_formatted(g, "typedef enum {\n");
    for (size_t i = 0; i < json_object_array_length(cases); i++) {
        json_object* node_class = json_object_array_get_idx(cases, i);
        String name = json_object_get_string(json_object_object_get(node_class, "name"));
        String capitalized = capitalize(name);
        shd_growy_append_formatted(g, "\t%s%s = 0x%x", enum_case_prefix, capitalized, (1 << i));
        shd_growy_append_formatted(g, ",\n");
        free((void*) capitalized);
    }
    shd_growy_append_formatted(g, "} %s;\n\n", enum_type_name);
}

void generate_bit_enum_classifier(Growy* g, String fn_name, String enum_type_name, String enum_case_prefix, String src_type_name, String src_case_prefix, String src_case_suffix, json_object* cases) {
    shd_growy_append_formatted(g, "%s %s(%s tag) {\n", enum_type_name, fn_name, src_type_name);
    shd_growy_append_formatted(g, "\tswitch (tag) { \n");
    assert(json_object_get_type(cases) == json_type_array);
    for (size_t i = 0; i < json_object_array_length(cases); i++) {
        json_object* node = json_object_array_get_idx(cases, i);
        String name = json_object_get_string(json_object_object_get(node, "name"));
        shd_growy_append_formatted(g, "\t\tcase %s%s%s: \n", src_case_prefix, name, src_case_suffix);
        json_object* class = json_object_object_get(node, "class");
        switch (json_object_get_type(class)) {
            case json_type_null:shd_growy_append_formatted(g, "\t\t\treturn 0;\n");
                break;
            case json_type_string: {
                String cap = capitalize(json_object_get_string(class));
                shd_growy_append_formatted(g, "\t\t\treturn %s%s;\n", enum_case_prefix, cap);
                free((void*) cap);
                break;
            }
            case json_type_array: {
                shd_growy_append_formatted(g, "\t\t\treturn ");
                for (size_t j = 0; j < json_object_array_length(class); j++) {
                    if (j > 0)
                        shd_growy_append_formatted(g, " | ");
                    String cap = capitalize(json_object_get_string(json_object_array_get_idx(class, j)));
                    shd_growy_append_formatted(g, "%s%s", enum_case_prefix, cap);
                    free((void*) cap);
                }
                shd_growy_append_formatted(g, ";\n");
                break;
            }
            case json_type_boolean:
            case json_type_double:
            case json_type_int:
            case json_type_object:
                shd_error_print("Invalid datatype for a node's 'class' attribute");
                break;
        }
    }
    shd_growy_append_formatted(g, "\t\tdefault: assert(false);\n");
    shd_growy_append_formatted(g, "\t}\n");
    shd_growy_append_formatted(g, "\tSHADY_UNREACHABLE;\n");
    shd_growy_append_formatted(g, "}\n");
}