#ifndef SHADY_PRINTER_H
#define SHADY_PRINTER_H

#include <stddef.h>

typedef struct Printer_ Printer;
typedef struct Growy_ Growy;

Printer* shd_new_printer_from_file(void* FILE);
Printer* shd_new_printer_from_growy(Growy* g);
void shd_destroy_printer(Printer* p);

Printer* shd_print(Printer*, const char*, ...);
void shd_newline(Printer* p);
void shd_printer_indent(Printer* p);
void shd_printer_deindent(Printer* p);
void shd_printer_flush(Printer* p);
void shd_printer_escape(Printer* p, const char*);
void shd_printer_unescape(Printer* p, const char*);

const char* shd_printer_growy_unwrap(Printer* p);
Growy* shd_new_growy(void);
#define shd_helper_format_string(f, ...) printer_growy_unwrap(cunk_print(cunk_open_growy_as_printer(cunk_new_growy()), (f), __VA_ARGS__))

#endif
