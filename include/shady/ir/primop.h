#ifndef SHADY_IR_PRIMOP_H
#define SHADY_IR_PRIMOP_H

#include "shady/ir/grammar.h"

String get_primop_name(Op op);
bool has_primop_got_side_effects(Op op);

#endif
