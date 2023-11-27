#include "generator.h"

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

void generate(Growy* g, Data data) {
    generate_header(g, data);

    json_object* primops = json_object_object_get(data.shd, "prim-ops");
    generate_primops_names_array(g, primops);
    generate_primops_side_effects_array(g, primops);

    generate_bit_enum_classifier(g, "get_primop_class", "OpClass", "Oc", "Op", "", "_op", primops);
}
