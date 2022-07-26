#ifndef SHADY_PORTABILITY
#define SHADY_PORTABILITY

#include <stdlib.h>

#include <assert.h>

inline static void check_c11() {
    static_assert(_CRT_HAS_C11, "C11 support is required to build shady.");
}

#ifdef _MSC_VER
    #define SHADY_UNUSED
    #define LARRAY(T, name, size) T* name = alloca(sizeof(T) * (size))
    #define alloca _alloca
    #define SHADY_FALLTHROUGH
    // It's mid 2022, and this typedef is missing from <stdalign.h>
    // MSVC is not a real C11 compiler.
    typedef long long max_align_t;
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
