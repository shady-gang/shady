#ifndef SHADY_UTIL
#define SHADY_UTIL

#include <stddef.h>
#include <stdbool.h>

size_t shd_apply_escape_codes(const char* src, size_t size, char* dst);
size_t shd_unapply_escape_codes(const char* src, size_t size, char* dst);

bool shd_read_file(const char* filename, size_t* size, char** output);
bool shd_write_file(const char* filename, size_t size, const char* data);

typedef struct Arena_ Arena;
char* shd_format_string_arena(Arena* arena, const char* str, ...);
char* shd_format_string_new(const char* str, ...);
bool shd_string_starts_with(const char* string, const char* prefix);
bool shd_string_ends_with(const char* string, const char* suffix);

char* shd_strip_path(const char*);

void shd_configure_int_flag_in_list(const char* str, const char* flag_name, int* flag_value);
void shd_configure_bool_flag_in_list(const char* str, const char* flag_name, bool* flag_value);

#endif
