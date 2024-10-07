#ifndef SHADY_IR_DEBUG_H
#define SHADY_IR_DEBUG_H

#include "shady/ir/base.h"

/// Get the name out of a global variable, function or constant
String shd_get_value_name_safe(const Node* v);
String shd_get_value_name_unsafe(const Node* v);
void shd_set_value_name(const Node* var, String name);

#endif
