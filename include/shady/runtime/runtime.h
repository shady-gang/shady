#ifndef SHD_RUNTIME
#define SHD_RUNTIME

#include "shady/ir/base.h"

/// Turns a constant into an array of actual bytes
void shd_rt_materialize_constant_at(void* target, const Node* value);

/// Turns a constant into an array of actual bytes
/// only returns the size if data is null
/// data must be at least as big as size
void shd_rt_materialize_constant(const Node* value, size_t* size, void* data);

#endif
