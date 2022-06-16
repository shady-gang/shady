#ifndef SHADY_PORTABILITY
#define SHADY_PORTABILITY

#include <stdlib.h>

#ifdef _MSC_VER
    #define SHADY_UNUSED
    #define LARRAY(T, name, size) T* name = alloca(sizeof(T) * (size))
    #define alloca _alloca
    #define SHADY_FALLTHROUGH
#else
    #ifdef USE_VLAS
        #define LARRAY(T, name, size) T name[size]
    #else
        #define LARRAY(T, name, size) T* name = alloca(sizeof(T) * (size))
    #endif
    #define SHADY_UNUSED __attribute__((unused))
    #define SHADY_FALLTHROUGH __attribute__((fallthrough));
#endif

#endif
