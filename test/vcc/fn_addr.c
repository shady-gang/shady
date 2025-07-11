#include <stddef.h>

size_t self() {
    return (size_t) &self;
}
