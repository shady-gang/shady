#ifndef SHADY_LOG_H
#define SHADY_LOG_H

#include <stdio.h>

typedef struct Node_ Node;
typedef struct Module_ Module;

typedef enum LogLevel_ {
    DEBUG,
    INFO,
    WARN,
    ERROR
} LogLevel;

LogLevel get_log_level();
void set_log_level(LogLevel);
void log_string(LogLevel level, const char* format, ...);
void log_node(LogLevel level, const Node* node);
void log_module(LogLevel level, Module*);

#define debug_print(...) log_string(DEBUG, __VA_ARGS__)
#define debug_node(n)    log_node(DEBUG, n)
#define debug_module(n)  log_module(DEBUG, n)

#define info_print(...)  log_string(INFO, __VA_ARGS__)
#define info_node(n)     log_node(INFO, n)
#define info_module(n)   log_module(INFO, n)

#define warn_print(...)  log_string(WARN, __VA_ARGS__)
#define warn_node(n)     log_node(WARN, n)
#define warn_module(n)   log_module(WARN, n)

#define error_print(...) log_string(ERROR, __VA_ARGS__)
#define error_node(n)    log_node(ERROR, n)
#define error_module(n)  log_module(ERROR, n)

#ifdef _MSC_VER
#define SHADY_UNREACHABLE __assume(0)
#else
#define SHADY_UNREACHABLE __builtin_unreachable()
#endif

#define SHADY_NOT_IMPLEM {    \
  error("not implemented\n"); \
  SHADY_UNREACHABLE;          \
}

#define error(...) {                                        \
  fprintf (stderr, "Error at %s:%d: ", __FILE__, __LINE__); \
  fprintf (stderr, __VA_ARGS__);                            \
  fprintf (stderr, "\n");                                   \
  error_die();                                              \
  SHADY_UNREACHABLE;                                        \
}

void error_die();

#endif
