#ifndef SHADY_IR_DEBUG_H
#define SHADY_IR_DEBUG_H

#include "shady/ir/base.h"
#include "shady/ir/builder.h"

/// Get the name out of a global variable, function or constant
String shd_get_node_name_safe(const Node* v);
String shd_get_node_name_unsafe(const Node* node);
void shd_set_debug_name(const Node* var, String name);

void shd_bld_comment(BodyBuilder* bb, String str);
void shd_bld_debug_printf(BodyBuilder* bb, String pattern, Nodes args);

#endif
