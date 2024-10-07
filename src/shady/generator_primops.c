#include "generator.h"

static void generate_primops_names_array(Growy* g, json_object* primops) {
    shd_growy_append_string(g, "const char* primop_names[] = {\n");

    for (size_t i = 0; i < json_object_array_length(primops); i++) {
        json_object* node = json_object_array_get_idx(primops, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        shd_growy_append_formatted(g, "\"%s\",", name);
    }

    shd_growy_append_string(g, "\n};\n");
}

static void generate_primops_side_effects_array(Growy* g, json_object* primops) {
    shd_growy_append_string(g, "const bool primop_side_effects[] = {\n");

    for (size_t i = 0; i < json_object_array_length(primops); i++) {
        json_object* node = json_object_array_get_idx(primops, i);

        String name = json_object_get_string(json_object_object_get(node, "name"));
        assert(name);

        bool side_effects = json_object_get_boolean(json_object_object_get(node, "side-effects"));
        if (side_effects)
            shd_growy_append_string(g, "true, ");
        else
            shd_growy_append_string(g, "false, ");
    }

    shd_growy_append_string(g, "\n};\n");
}

void generate(Growy* g, json_object* shd) {
    generate_header(g, shd);

    json_object* primops = json_object_object_get(shd, "prim-ops");
    generate_primops_names_array(g, primops);
    generate_primops_side_effects_array(g, primops);

    generate_bit_enum_classifier(g, "shd_get_primop_class", "OpClass", "Oc", "Op", "", "_op", primops);
}
