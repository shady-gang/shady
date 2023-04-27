#ifndef SHADY_PORTABILITY
#define SHADY_PORTABILITY

#include <stddef.h>
#include <stdlib.h>
#ifdef _MSC_VER
#include <malloc.h>
#endif

#include <assert.h>

static_assert(__STDC_VERSION__ >= 201112L, "C11 support is required to build shady.");

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

static inline void* alloc_aligned(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static inline void free_aligned(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void platform_specific_terminal_init_extras();

#endif
