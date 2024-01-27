#include "generator.h"

#include "util.h"

enum {
    ArgSelf = 0,
    ArgDstFile,
    ArgSpirvGrammarSearchPathBegins
};

void import_spirv_defs(json_object* src, json_object* dst) {
    json_object* spv = json_object_new_object();
    json_object_object_add(spv, "major_version", json_object_object_get(src, "major_version"));
    json_object_object_add(spv, "minor_version", json_object_object_get(src, "minor_version"));
    json_object_object_add(spv, "revision", json_object_object_get(src, "revision"));
    json_object_object_add(dst, "spv", spv);
}

int main(int argc, char** argv) {
    assert(argc > ArgSpirvGrammarSearchPathBegins);

    //char* mode = argv[ArgGeneratorFn];
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

    JsonFile json_file;
    read_file(spv_core_json_path, &json_file.size, &json_file.contents);
    json_file.root = json_tokener_parse_ex(tokener, json_file.contents, json_file.size);
    json_err = json_tokener_get_error(tokener);
    if (json_err != json_tokener_success) {
        error("Json tokener error while parsing %s:\n %s\n", spv_core_json_path, json_tokener_error_desc(json_err));
    }

    info_print("Correctly opened json file: %s\n", spv_core_json_path);

    json_object* output = json_object_new_object();

    import_spirv_defs(json_file.root, output);

    Growy* g = new_growy();
    growy_append_string(g, json_object_to_json_string(output));
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
    free(json_file.contents);
    json_object_put(json_file.root);
    json_tokener_free(tokener);
    free(spv_core_json_path);
}
