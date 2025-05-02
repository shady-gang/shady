#include "portability.h"

#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

// Fix for allowing terminal colors on MINGW64
// See: https://gist.github.com/fleroviux/8343879d95a72140274535dc207f467d
#if defined(__MINGW32__)

// Include windows.h and if this macro is defined we apply the fix
// Otherwise we assume we are running one some older version of windows and the fix can't be applied...
#include <windows.h>
#if defined(ENABLE_VIRTUAL_TERMINAL_PROCESSING)
#define NEED_COLOR_FIX
#endif

#endif

void shd_platform_specific_terminal_init_extras(void) {
#ifdef NEED_COLOR_FIX
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (handle != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(handle, &mode)) {
            mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(handle, mode);
        }
    }
#endif
}

#include <stdint.h>
#if defined(__MINGW64__) | defined(__MINGW32__)
#include <pthread.h>
uint64_t shd_get_time_nano() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
}
#else
#include <time.h>
uint64_t shd_get_time_nano(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec * 1000000000 + t.tv_nsec;
}
#endif

#ifdef WIN32
#include <windows.h>
#elif __APPLE__
#include <mach-o/dyld.h>
#include <limits.h>
#else
#include <unistd.h>
#include <stdio.h>
#endif
const char* shd_get_executable_location(void) {
    size_t len = 256;
    char* buf = calloc(len + 1, 1);
#ifdef WIN32
    size_t final_len = GetModuleFileNameA(NULL, buf, len);
#elif __APPLE__
    uint32_t final_len = len;
    _NSGetExecutablePath(buf, &final_len);
#else
    size_t final_len = readlink("/proc/self/exe", buf, len);
#endif
    assert(final_len <= len);
    return buf;
}

#ifdef __USE_POSIX
#include "signal.h"
void shd_breakpoint(SHADY_UNUSED const char* message) {
    raise(SIGTRAP);
}
#elif WIN32
void shd_breakpoint(SHADY_UNUSED const char* message) {
    __debugbreak();
}
#else
void shd_breakpoint(SHADY_UNUSED const char* message) {
    exit(666);
}
#endif
