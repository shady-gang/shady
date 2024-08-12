#ifndef SHADY_PASS_H
#define SHADY_PASS_H

#include "shady/config.h"
#include "rewrite.h"

typedef Module* (RewritePass)(const CompilerConfig* config, Module* src);
typedef bool (OptPass)(const CompilerConfig* config, Module** m);

#endif

