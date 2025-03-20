#include "log.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static LogLevel default_log_level = INFO;
static bool log_level_init = false;
LogLevel shady_log_level = INFO;

LogLevel shd_log_get_level(void) {
    if (!log_level_init) {
        shady_log_level = default_log_level;
        char* env_level = getenv("SHD_LOG_LEVEL");
        if (env_level) {
            if (strcmp(env_level, "ERROR") == 0)
                shady_log_level = ERROR;
            if (strcmp(env_level, "INFO") == 0)
                shady_log_level = INFO;
            if (strcmp(env_level, "WARN") == 0)
                shady_log_level = WARN;
            if (strcmp(env_level, "INFO") == 0)
                shady_log_level = INFO;
            if (strcmp(env_level, "DEBUG") == 0)
                shady_log_level = DEBUG;
            if (strcmp(env_level, "DEBUGV") == 0)
                shady_log_level = DEBUGVV;
            if (strcmp(env_level, "DEBUGVV") == 0)
                shady_log_level = DEBUGVV;
        }
        log_level_init = true;
    }
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
