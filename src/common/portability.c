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

void platform_specific_terminal_init_extras() {
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

#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <stdio.h>
#endif
const char* get_executable_location(void) {
    size_t len = 256;
    char* buf = calloc(len + 1, 1);
#ifdef WIN32
    size_t final_len = GetModuleFileNameA(NULL, buf, len);
#else
    size_t final_len = readlink("/proc/self/exe", buf, len);
#endif
    assert(final_len <= len);
    return buf;
}