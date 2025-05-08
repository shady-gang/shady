#ifndef SHADY_CLI_H
#define SHADY_CLI_H

#include <string.h>

#define PARSE_TOGGLE_OPTION(f, name) \
if (strcmp(argv[i], "--no-"#name) == 0) { \
    f = false; argv[i] = NULL; continue; \
} else if (strcmp(argv[i], "--"#name) == 0) { \
    f = true; argv[i] = NULL; continue; \
}

void shd_pack_remaining_args(int* pargc, char** argv);

#endif
