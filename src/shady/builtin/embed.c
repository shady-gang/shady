#include <stdio.h>
#include <string.h>
#include <stddef.h>

typedef enum {
    STRING, BIN
} Mode;

int main(int argc, char** argv) {
    if (argc != 5) {
        goto usage_fail;
    }

    Mode m;
    if (strcmp(argv[1], "string") == 0)
        m = STRING;
    else if (strcmp(argv[1], "bin") == 0)
        m = BIN;
    else
        goto usage_fail;

    const char* object_name = argv[2];

    FILE* src = fopen(argv[3], "rb");
    FILE* dst = fopen(argv[4], "wb");
    if (!src || !dst)
        goto io_fail;

    fseek(src, 0, SEEK_END);
    size_t size = ftell(src);
    fseek(src, 0, 0);

    printf("source file is %zd bytes long\n", size);

    size_t pos = 0;
    char buffer[16384];
    size_t read;

    fprintf(dst, "const char %s[] = ", object_name);
    switch (m) {
        case STRING: {
            fprintf(dst, "\"");
            while (pos < size) {
                read = fread(buffer, sizeof(char), sizeof(buffer) / sizeof(char), src);
                for (size_t i = 0; i < read; i++) {
                    char c = buffer[i];
                    switch (c) {
                        case '"':
                            fprintf(dst, "\\\"");
                            break;
                        case '\n':
                            fprintf(dst, "\\n\"\n\"");
                            break;
                        default:
                            fprintf(dst, "%c", c);
                            break;
                    }
                    pos++;
                }
                if (read == 0)
                    break;
            }
            fprintf(dst, "\"");
            break;
        }
        case BIN: {
            fprintf(dst, "{");
            while (pos < size) {
                read = fread(buffer, sizeof(char), sizeof(buffer) / sizeof(char), src);
                for (size_t i = 0; i < read; i++) {
                    char c = buffer[i];
                    fprintf(dst, "%d, ", c);
                    pos++;
                }
                if (read == 0)
                    break;
            }
            fprintf(dst, "}");
            break;
        }
    }
    fprintf(dst, ";\n");
    fclose(dst);

    return 0;

    io_fail:
    printf("i/o failure\n");
    return 2;

    usage_fail:
    printf("Usage: embedder [string | bin] name src_file dst_file\n");
    return 1;
}
