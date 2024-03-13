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
        read_file(path, &json_files[i].size, &json_files[i].contents);
        json_files[i].root = json_tokener_parse_ex(tokener, json_files[i].contents, json_files[i].size);
        json_err = json_tokener_get_error(tokener);
        if (json_err != json_tokener_success) {
            error("Json tokener error while parsing %s:\n %s\n", path, json_tokener_error_desc(json_err));
        }

        info_print("Correctly opened json file: %s\n", path);
    }
    Growy* g = new_growy();

    json_object* src = json_object_new_object();

    for (size_t i = 0; i < inputs_count; i++) {
        json_apply_object(src, json_files[i].root);
    }

    preprocess(src);
    generate(g, src);

    size_t final_size = growy_size(g);
    growy_append_bytes(g, 1, (char[]) { 0 });
    char* generated = growy_deconstruct(g);
    debug_print("debug: %s\n", generated);
    if (!write_file(dst_file, final_size, generated)) {
        error_print("Failed to write file '%s'\n", dst_file);
        error_die();
    }
    free(generated);
    for (size_t i = 0; i < inputs_count; i++) {
        free(json_files[i].contents);
        json_object_put(json_files[i].root);
    }
    json_object_put(src);
    json_tokener_free(tokener);
}
