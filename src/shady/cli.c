#include "shady/cli.h"

#include "portability.h"

void shd_pack_remaining_args(int* pargc, char** argv) {
    LARRAY(char*, nargv, *pargc);
    int nargc = 0;
    for (size_t i = 0; i < *pargc; i++) {
        if (argv[i] == NULL) continue;
        nargv[nargc++] = argv[i];
    }
    memcpy(argv,nargv, sizeof(char*) * nargc);
    *pargc = nargc;
}