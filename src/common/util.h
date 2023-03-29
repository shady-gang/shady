#ifndef SHADY_UTIL
#define SHADY_UTIL

#include <stddef.h>
#include <stdbool.h>

bool read_file(const char* filename, size_t* size, unsigned char** output);
bool write_file(const char* filename, size_t size, const unsigned char* data);

#endif
