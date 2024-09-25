#include "log.h"

#include <stdio.h>

LogLevel shady_log_level = INFO;

LogLevel shd_log_get_level(void) {
    return shady_log_level;
}

void shd_log_set_level(LogLevel l) {
    shady_log_level = l;
}

void shd_log_fmt_va_list(LogLevel level, const char* format, va_list args) {
    if (level <= shady_log_level)
        vfprintf(stderr, format, args);
}

void shd_log_fmt(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    shd_log_fmt_va_list(level, format, args);
    va_end(args);
}
