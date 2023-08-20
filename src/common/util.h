#ifndef SHADY_UTIL
#define SHADY_UTIL

#include <stddef.h>
#include <stdbool.h>

size_t apply_escape_codes(const char* src, size_t og_len, char* dst);

bool read_file(const char* filename, size_t* size, unsigned char** output);
bool write_file(const char* filename, size_t size, const unsigned char* data);

#endif
