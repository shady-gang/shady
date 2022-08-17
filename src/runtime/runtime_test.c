#include "shady/runtime.h"

#include "log.h"

#include <stdlib.h>
#include <stdio.h>

static const char* shader =
"// trivial function that returns its argument\n"
"@EntryPoint(\"compute\") fn main() {\n"
"    return;\n"
"}";

int main(int argc, const char* argv[]) {
    set_log_level(DEBUG);
    info_print("Shady runtime test starting...\n");

    RuntimeConfig config = {
        .use_validation = true,
    };
    Runtime* runtime = initialize_runtime(config);
    Program* program = load_program(runtime, shader);
    shutdown_runtime(runtime);
}
