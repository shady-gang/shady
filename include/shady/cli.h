#ifndef SHADY_CLI_H
#define SHADY_CLI_H

#include <string.h>
#include <stdbool.h>

#define PARSE_TOGGLE_OPTION(f, name) \
if (strcmp(argv[i], "--no-"#name) == 0) { \
    f = false; argv[i] = NULL; continue; \
} else if (strcmp(argv[i], "--"#name) == 0) { \
    f = true; argv[i] = NULL; continue; \
}

void shd_pack_remaining_args(int* pargc, char** argv);

bool shd_is_arg_help(const char* arg);
// return 'true' if --help was amongst the passed arguments, also removes it if asked
bool shd_parse_help(int* pargc, char** argv, bool remove);

#endif
