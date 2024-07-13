#include "generator.h"

#include "util.h"
#include "list.h"

enum {
    ArgSelf = 0,
    ArgDstFile,
    ArgImportsFile,
    ArgSpirvGrammarSearchPathBegins
};

static String sanitize_node_name(String name) {
    char* tmpname = NULL;
    tmpname = calloc(strlen(name) + 1, 1);
    bool is_type = false;
    if (string_starts_with(name, "OpType")) {
        memcpy(tmpname, name + 6, strlen(name) - 6);
        is_type = true;
    } else if (string_starts_with(name, "Op"))
       memcpy(tmpname, name + 2, strlen(name) - 2);
    else
       memcpy(tmpname, name, strlen(name));

    if (is_type)
        memcpy(tmpname + strlen(tmpname), "Type", 4);

    return tmpname;
}

static String sanitize_field_name(String name) {
    char* tmpname = NULL;
    tmpname = calloc(strlen(name) + 1, 1);
    if (name[0] == '\'') {
        memcpy(tmpname, name + 1, strlen(name) - 2);
        name = tmpname;
    } else {
        memcpy(tmpname, name, strlen(name));
    }
    for (size_t i = 0; i < strlen(tmpname); i++) {
        if (tmpname[i] == ' ')
            tmpname[i] = '_';
        else
            tmpname[i] = tolower(tmpname[i]);
    }
    return tmpname;
}

static void copy_object(json_object* dst, json_object* src, String name, String copied_name) {
    json_object* o = json_object_object_get(src, name);
    json_object_get(o);
    json_object_object_add(dst, copied_name ? copied_name : name, o);
}

void apply_instruction_filter(json_object* filter, json_object* instruction, json_object* instantiated_filter, struct List* pending) {
    switch (json_object_get_type(filter)) {
        case json_type_array: {
            for (size_t i = 0; i < json_object_array_length(filter); i++) {
                apply_instruction_filter(json_object_array_get_idx(filter, i), instruction, instantiated_filter, pending);
            }
            break;
        }
        case json_type_object: {
            json_object* filter_name = json_object_object_get(filter, "filter-name");
            if (filter_name) {
                assert(json_object_get_type(filter_name) == json_type_object);
                String name = json_object_get_string(json_object_object_get(instruction, "opname"));
                bool found = false;
                json_object_object_foreach(filter_name, match_name, subfilter) {
                    if (strcmp(name, match_name) == 0) {
                        found = true;
                        append_list(json_object*, pending, subfilter);
                    }
                }
                if (!found)
                    return;
            }

            json_apply_object(instantiated_filter, filter);
            /*json_object_object_foreach(filter, proprerty, value) {
                json_object_get(value);
                json_object_object_add(instantiated_filter, proprerty, value);
            }*/
            break;
        }
        default: error("Filters need to be arrays or objects");
    }
}

json_object* apply_instruction_filters(json_object* filter, json_object* instruction) {
    json_object* instantiated_filter = json_object_new_object();
    struct List* pending = new_list(json_object*);
    apply_instruction_filter(filter, instruction, instantiated_filter, pending);
    while(entries_count_list(pending) > 0) {
        json_object* pending_filter = read_list(json_object*, pending)[0];
        remove_list(json_object*, pending, 0);
        apply_instruction_filter(pending_filter, instruction, instantiated_filter, pending);
        continue;
    }
    destroy_list(pending);
    return instantiated_filter;
}

void apply_operand_filter(json_object* filter, json_object* operand, json_object* instantiated_filter, struct List* pending) {
    //fprintf(stderr, "applying %s\n", json_object_to_json_string(filter));
    switch (json_object_get_type(filter)) {
        case json_type_array: {
            for (size_t i = 0; i < json_object_array_length(filter); i++) {
                apply_operand_filter(json_object_array_get_idx(filter, i), operand, instantiated_filter, pending);
            }
            break;
        }
        case json_type_object: {
            json_object* filter_name = json_object_object_get(filter, "filter-name");
            if (filter_name) {
                assert(json_object_get_type(filter_name) == json_type_object);
                String name = json_object_get_string(json_object_object_get(operand, "name"));
                if (!name)
                    name = "";
                bool found = false;
                json_object_object_foreach(filter_name, match_name, subfilter) {
                    if (strcmp(name, match_name) == 0) {
                        found = true;
                        append_list(json_object*, pending, subfilter);
                    }
                }
                if (!found)
                    return;
            }
            json_object* filter_kind = json_object_object_get(filter, "filter-kind");
            if (filter_kind) {
                assert(json_object_get_type(filter_kind) == json_type_object);
                String kind = json_object_get_string(json_object_object_get(operand, "kind"));
                if (!kind)
                    kind = "";
                bool found = false;
                json_object_object_foreach(filter_kind, match_name, subfilter) {
                    if (strcmp(kind, match_name) == 0) {
                        found = true;
                        append_list(json_object*, pending, subfilter);
                    }
                }
                if (!found)
                    return;
            }

            json_apply_object(instantiated_filter, filter);
            break;
        }
        default: error("Filters need to be arrays or objects");
    }
}

json_object* apply_operand_filters(json_object* filter, json_object* operand) {
    //fprintf(stderr, "building filter for %s\n", json_object_to_json_string(operand));
    json_object* instantiated_filter = json_object_new_object();
    struct List* pending = new_list(json_object*);
    apply_operand_filter(filter, operand, instantiated_filter, pending);
    while(entries_count_list(pending) > 0) {
        json_object* pending_filter = read_list(json_object*, pending)[0];
        remove_list(json_object*, pending, 0);
        apply_operand_filter(pending_filter, operand, instantiated_filter, pending);
        continue;
    }
    destroy_list(pending);
    //fprintf(stderr, "done: %s\n", json_object_to_json_string(instantiated_filter));
    return instantiated_filter;
}

json_object* import_operand(json_object* operand, json_object* instruction_filter) {
    String kind = json_object_get_string(json_object_object_get(operand, "kind"));
    assert(kind);
    String name = json_object_get_string(json_object_object_get(operand, "name"));
    if (!name)
        name = kind;

    json_object* operand_filters = json_object_object_get(instruction_filter, "operand-filters");
    assert(operand_filters);
    json_object* filter = apply_operand_filters(operand_filters, operand);

    String import_property = json_object_get_string(json_object_object_get(filter, "import"));
    if (!import_property || (strcmp(import_property, "no") == 0)) {
        json_object_put(filter);
        return NULL;
    } else if (strcmp(import_property, "yes") != 0) {
        error("a filter's 'import' property needs to be 'yes' or 'no'")
    }

    json_object* field = json_object_new_object();

    const char* field_name = sanitize_field_name(name);
    json_object_object_add(field, "name", json_object_new_string(field_name));
    free((void*) field_name);

    json_object* insert = json_object_object_get(filter, "overlay");
    if (insert) {
        json_apply_object(field, insert);
    }
    json_object_put(filter);

    return field;
}

json_object* import_filtered_instruction(json_object* instruction, json_object* filter) {
    String name = json_object_get_string(json_object_object_get(instruction, "opname"));
    assert(name && strlen(name) > 2);

    String import_property = json_object_get_string(json_object_object_get(filter, "import"));
    if (!import_property || (strcmp(import_property, "no") == 0)) {
        return NULL;
    } else if (strcmp(import_property, "yes") != 0) {
        error("a filter's 'import' property needs to be 'yes' or 'no'")
    }
    String node_name = sanitize_node_name(name);

    json_object* node = json_object_new_object();
    json_object_object_add(node, "name", json_object_new_string(node_name));
    copy_object(node, instruction, "opcode", "spirv-opcode");

    json_object* insert = json_object_object_get(filter, "overlay");
    if (insert) {
        json_apply_object(node, insert);
    }

    json_object* operands = json_object_object_get(instruction, "operands");
    assert(operands);
    json_object* ops = json_object_new_array();
    for (size_t i = 0; i < json_object_array_length(operands); i++) {
        json_object* operand = json_object_array_get_idx(operands, i);
        json_object* field = import_operand(operand, filter);
        if (field)
            json_object_array_add(ops, field);
    }

    if (json_object_array_length(ops) > 0)
        json_object_object_add(node, "ops", ops);
    else
        json_object_put(ops);

    free((void*) node_name);
    return node;
}

void import_spirv_defs(json_object* imports, json_object* src, json_object* dst) {
    json_object* spv = json_object_new_object();
    json_object_object_add(dst, "spv", spv);
    copy_object(spv, src, "major_version", NULL);
    copy_object(spv, src, "minor_version", NULL);
    copy_object(spv, src, "revision", NULL);

    // import instructions
    json_object* filters = json_object_object_get(imports, "instruction-filters");
    json_object* nodes = json_object_new_array();
    json_object_object_add(dst, "nodes", nodes);
    json_object* instructions = json_object_object_get(src, "instructions");
    //assert(false);
    for (size_t i = 0; i < json_object_array_length(instructions); i++) {
        json_object* instruction = json_object_array_get_idx(instructions, i);

        json_object* filter = apply_instruction_filters(filters, instruction);
        json_object* result = import_filtered_instruction(instruction, filter);
        if (result) {
            json_object_array_add(nodes, result);
        }
        json_object_put(filter);
    }
}

int main(int argc, char** argv) {
    assert(argc > ArgSpirvGrammarSearchPathBegins);

    char* dst_file = argv[ArgDstFile];
    // search the include path for spirv.core.grammar.json
    char* spv_core_json_path = NULL;
    for (size_t i = ArgSpirvGrammarSearchPathBegins; i < argc; i++) {
        char* path = format_string_new("%s/spirv/unified1/spirv.core.grammar.json", argv[i]);
        info_print("trying path %s\n", path);
        FILE* f = fopen(path, "rb");
        if (f) {
            spv_core_json_path = path;
            fclose(f);
            break;
        }
        free(path);
    }

    if (!spv_core_json_path)
        abort();

    json_tokener* tokener = json_tokener_new_ex(32);
    enum json_tokener_error json_err;

    typedef struct {
        size_t size;
        char* contents;
        json_object* root;
    } JsonFile;

    JsonFile imports;
    read_file(argv[ArgImportsFile], &imports.size, &imports.contents);
    imports.root = json_tokener_parse_ex(tokener, imports.contents, imports.size);
    json_err = json_tokener_get_error(tokener);
    if (json_err != json_tokener_success) {
        error("Json tokener error while parsing %s:\n %s\n", argv[ArgImportsFile], json_tokener_error_desc(json_err));
    }

    JsonFile spirv;
    read_file(spv_core_json_path, &spirv.size, &spirv.contents);
    spirv.root = json_tokener_parse_ex(tokener, spirv.contents, spirv.size);
    json_err = json_tokener_get_error(tokener);
    if (json_err != json_tokener_success) {
        error("Json tokener error while parsing %s:\n %s\n", spv_core_json_path, json_tokener_error_desc(json_err));
    }

    info_print("Correctly opened json file: %s\n", spv_core_json_path);

    json_object* output = json_object_new_object();

    import_spirv_defs(imports.root, spirv.root, output);

    Growy* g = new_growy();
    growy_append_string(g, json_object_to_json_string_ext(output, JSON_C_TO_STRING_PRETTY));
    json_object_put(output);

    size_t final_size = growy_size(g);
    growy_append_bytes(g, 1, (char[]) { 0 });
    char* generated = growy_deconstruct(g);
    debug_print("debug: %s\n", generated);
    if (!write_file(dst_file, final_size, generated)) {
        error_print("Failed to write file '%s'\n", dst_file);
        error_die();
    }
    free(generated);
    free(spirv.contents);
    json_object_put(spirv.root);
    free(imports.contents);
    json_object_put(imports.root);
    json_tokener_free(tokener);
    free(spv_core_json_path);
}
