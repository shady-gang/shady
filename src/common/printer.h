#ifndef SHADY_PRINTER_H
#define SHADY_PRINTER_H

#include <stddef.h>

typedef struct Printer_ Printer;
typedef struct Growy_ Growy;

Printer* open_file_as_printer(void* FILE);
Printer* open_growy_as_printer(Growy*);
void destroy_printer(Printer*);

Printer* print(Printer*, const char*, ...);
void newline(Printer* p);
void indent(Printer* p);
void deindent(Printer* p);
void flush(Printer*);

/// Prints a size using appropriate KiB/MiB/GiB suffixes
void print_size_suffix(Printer*, size_t, int extra);

const char* printer_growy_unrwap(Printer* p);
Growy* new_growy();
#define format_string(f, ...) printer_growy_unrwap(cunk_print(cunk_open_growy_as_printer(cunk_new_growy()), (f), __VA_ARGS__))

const char* replace_string(const char* source, const char* match, const char* replace_with);

#endif
