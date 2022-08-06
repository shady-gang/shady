#include "shady/runtime.h"

#include "../log.h"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, const char* argv[]) {
    set_log_level(DEBUG);
    info_print("Shady runtime test starting...\n");

    Runtime* runtime = initialize_runtime();
    shutdown_runtime(runtime);
}
