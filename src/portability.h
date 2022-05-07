#ifndef SHADY_PORTABILITY
#define SHADY_PORTABILITY

#include <stdlib.h>


#ifdef _MSC_VER
#define SHADY_UNUSED
#define LARRAY(T, name, size) T* name = alloca(sizeof(T) * (size))
#define alloca _alloca
#else
#define LARRAY(T, name, size) T name[size]
#define SHADY_UNUSED __attribute__((unused))
#endif

#endif
