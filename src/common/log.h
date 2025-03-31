#ifndef SHADY_LOG_H
#define SHADY_LOG_H

#include <stdio.h>
#include <stdarg.h>

typedef struct Node_ Node;
typedef struct Module_ Module;
typedef struct NodePrintConfig_ NodePrintConfig;

typedef enum LogLevel_ {
    ERROR,
    WARN,
    INFO,
    DEBUG,
    DEBUGV,
    DEBUGVV,
} LogLevel;

LogLevel shd_log_get_level(void);
void shd_log_set_level(LogLevel l);
void shd_log_fmt_va_list(LogLevel level, const char* format, va_list args);
void shd_log_fmt(LogLevel level, const char* format, ...);
void shd_log_node(LogLevel level, const Node* node);
void shd_log_node_config(LogLevel level, const Node* node, const NodePrintConfig*);
typedef struct CompilerConfig_ CompilerConfig;
void shd_log_module(LogLevel level, Module* mod);
void shd_log_module_config(LogLevel level, Module* mod, const NodePrintConfig*);

#define shd_debugvv_print(...) shd_log_fmt(DEBUGVV, __VA_ARGS__)
#define shd_debugv_print(...)  shd_log_fmt(DEBUGV, __VA_ARGS__)
#define shd_debug_print(...)   shd_log_fmt(DEBUG, __VA_ARGS__)
#define shd_info_print(...)    shd_log_fmt(INFO, __VA_ARGS__)
#define shd_warn_print(...)    shd_log_fmt(WARN, __VA_ARGS__)
#define shd_error_print(...)   shd_log_fmt(ERROR, __VA_ARGS__)

#define shd_debugvv_print_once(flag, ...) { static bool flag = false; if (!flag) { flag = true; shd_debugvv_print(__VA_ARGS__ ); } }
#define shd_debugv_print_once(flag, ...)  { static bool flag = false; if (!flag) { flag = true; shd_debugv_print(__VA_ARGS__ );  } }
#define shd_debug_print_once(flag, ...)   { static bool flag = false; if (!flag) { flag = true; shd_debug_print(__VA_ARGS__ );   } }
#define shd_info_print_once(flag, ...)    { static bool flag = false; if (!flag) { flag = true; shd_info_print(__VA_ARGS__ );    } }
#define shd_warn_print_once(flag, ...)    { static bool flag = false; if (!flag) { flag = true; shd_warn_print(__VA_ARGS__ );    } }
#define shd_error_print_once(flag, ...)   { static bool flag = false; if (!flag) { flag = true; shd_error_print(__VA_ARGS__ );   } }

#ifdef _MSC_VER
#define SHADY_UNREACHABLE __assume(0)
#else
#define SHADY_UNREACHABLE __builtin_unreachable()
#endif

#define SHADY_NOT_IMPLEM {    \
  error("not implemented\n"); \
  SHADY_UNREACHABLE;          \
}

#define shd_error(...) {                                    \
  fprintf (stderr, "Error at %s:%d: ", __FILE__, __LINE__); \
  fprintf (stderr, __VA_ARGS__);                            \
  fprintf (stderr, "\n");                                   \
  shd_error_die();                                          \
}

#define CHECK(x, failure_handler) { if (!(x)) { shd_error_print(#x " failed\n"); failure_handler; } }

#include <stdnoreturn.h>
noreturn void shd_error_die(void);

#endif
