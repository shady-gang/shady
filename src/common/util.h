#ifndef SHADY_UTIL
#define SHADY_UTIL

#include <stddef.h>
#include <stdbool.h>

bool read_file(const char* filename, size_t* size, unsigned char** output);

#endif
