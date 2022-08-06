#include "shady/runtime.h"

#include "../log.h"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, const char* argv[]) {
    set_log_level(DEBUG);
    info_print("Shady runtime test starting...\n");

    RuntimeConfig config = {
        .use_validation = true
    };
    Runtime* runtime = initialize_runtime(config);
    shutdown_runtime(runtime);
}
