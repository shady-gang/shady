#include "generator.h"

#include "util.h"
#include "portability.h"

void preprocess(json_object* root);
void generate(Growy* g, json_object* root);

enum {
    ArgSelf = 0,
    ArgDstFile,
    ArgFirstInput,
};

int main(int argc, char** argv) {
    assert(argc > ArgFirstInput);
    int inputs_count = argc - ArgFirstInput;
    char* dst_file = argv[ArgDstFile];

    json_tokener* tokener = json_tokener_new_ex(32);
    enum json_tokener_error json_err;

    typedef struct {
        size_t size;
        char* contents;
        json_object* root;
    } JsonFile;

    LARRAY(JsonFile, json_files, inputs_count);
    for (size_t i = 0; i < inputs_count; i++) {
        String path = argv[ArgFirstInput + i];
        shd_read_file(path, &json_files[i].size, &json_files[i].contents);
        json_files[i].root = json_tokener_parse_ex(tokener, json_files[i].contents, json_files[i].size);
        json_err = json_tokener_get_error(tokener);
        if (json_err != json_tokener_success) {
            shd_error("Json tokener error while parsing %s:\n %s\n", path, json_tokener_error_desc(json_err));
        }

        shd_info_print("Correctly opened json file: %s\n", path);
    }
    Growy* g = shd_new_growy();

    json_object* src = json_object_new_object();

    for (size_t i = 0; i < inputs_count; i++) {
        json_apply_object(src, json_files[i].root);
    }

    preprocess(src);
    generate(g, src);

    size_t final_size = shd_growy_size(g);
    shd_growy_append_bytes(g, 1, (char[]) { 0 });
    char* generated = shd_growy_deconstruct(g);
    shd_debug_print("debug: %s\n", generated);
    if (!shd_write_file(dst_file, final_size, generated)) {
        shd_error_print("Failed to write file '%s'\n", dst_file);
        shd_error_die();
    }
    free(generated);
    for (size_t i = 0; i < inputs_count; i++) {
        free(json_files[i].contents);
        json_object_put(json_files[i].root);
    }
    json_object_put(src);
    json_tokener_free(tokener);
}
