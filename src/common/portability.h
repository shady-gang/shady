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
    #define popen _popen
    #define pclose _pclose
    #define SHADY_FALLTHROUGH
    // It's mid 2022, and this typedef is missing from <stdalign.h>
    // MSVC is not a real C11 compiler.
    typedef double max_align_t;
#else
    #ifdef USE_VLAS
        #define LARRAY(T, name, size) T name[size]
    #else
        #define LARRAY(T, name, size) T* name = alloca(sizeof(T) * (size))
    #endif
    #define SHADY_UNUSED __attribute__((unused))
    #define SHADY_FALLTHROUGH __attribute__((fallthrough));
#endif

inline static size_t _shd_round_up(size_t a, size_t b) {
    size_t divided = (a + b - 1) / b;
    return divided * b;
}

static inline void* shd_alloc_aligned(size_t size, size_t alignment) {
    size = _shd_round_up(size, alignment);
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static inline void shd_free_aligned(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#include <stdint.h>
uint64_t shd_get_time_nano(void);
const char* shd_get_executable_location(void);

void shd_breakpoint(const char*);

void shd_platform_specific_terminal_init_extras(void);

#endif
