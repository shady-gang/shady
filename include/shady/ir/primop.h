#ifndef SHADY_IR_PRIMOP_H
#define SHADY_IR_PRIMOP_H

#include "shady/ir/grammar.h"

OpClass shd_get_primop_class(Op op);

String shd_get_primop_name(Op op);
bool shd_has_primop_got_side_effects(Op op);

#endif
