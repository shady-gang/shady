#include "util.h"
#include "printer.h"

#undef NDEBUG
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

const char escaped[] = "hi\nthis is a backslash\\, \tthis is a tab and this backspace character ends it all\b";
const char double_escaped[] = "hi\\nthis is a backslash\\\\, \\tthis is a tab and this backspace character ends it all\\b";

enum {
    Len = sizeof(escaped),
    DoubleLen = sizeof(double_escaped),
    MaxLen = DoubleLen
};

void test_escape_unescape_basic(void) {
    char output[MaxLen] = { 0 };

    printf("escaped: %s\n---------------------\n", escaped);
    printf("double_escaped: %s\n---------------------\n", double_escaped);
    shd_apply_escape_codes(double_escaped, DoubleLen, output);
    printf("shd_apply_escape_codes(double_escaped): %s\n---------------------\n", output);
    assert(strcmp(output, escaped) == 0);
    memset(output, 0, MaxLen);
    shd_unapply_escape_codes(escaped, Len, output);
    printf("shd_apply_escape_codes(escaped): %s\n---------------------\n", output);
    assert(strcmp(output, double_escaped) == 0);
}

void test_escape_printer(void) {
    Printer* p = shd_new_printer_from_growy(shd_new_growy());
    shd_printer_escape(p, double_escaped);
    const char* output = shd_printer_growy_unwrap(p);
    printf("shd_printer_escape(escaped): %s\n---------------------\n", output);
    assert(strlen(output) == Len - 1);
    assert(strcmp(output, escaped) == 0);
    free((char*) output);
}

void test_unescape_printer(void) {
    Printer* p = shd_new_printer_from_growy(shd_new_growy());
    shd_printer_unescape(p, escaped);
    const char* output = shd_printer_growy_unwrap(p);
    printf("shd_printer_unescape(escaped): %s\n---------------------\n", output);
    assert(strlen(output) == DoubleLen - 1);
    assert(strcmp(output, double_escaped) == 0);
    free((char*) output);
}

int main(int argc, char** argv) {
    assert(strlen(double_escaped) == DoubleLen - 1);
    assert(strlen(escaped) == Len - 1);

    test_escape_unescape_basic();
    test_escape_printer();
    test_unescape_printer();

    return 0;
}