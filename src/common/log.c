#include "log.h"

#include <stdio.h>
#include <stdarg.h>

LogLevel shady_log_level = INFO;

LogLevel get_log_level() {
    return shady_log_level;
}

void set_log_level(LogLevel l) {
    shady_log_level = l;
}

void log_string(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    if (level <= shady_log_level)
        vfprintf(stderr, format, args);
    va_end(args);
}
