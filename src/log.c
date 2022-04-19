#include "log.h"

#include <stdio.h>
#include <stdarg.h>

LogLevel log_level = ERROR;

void set_log_level(LogLevel l) {
    log_level = l;
}

void log_string(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    if (level >= log_level)
        vfprintf(stderr, format, args);
    va_end(args);
}
