#ifndef SHD_BE_DUMP_H
#define SHD_BE_DUMP_H

#include "shady/ir/base.h"
#include "shady/ir/module.h"

#include <stdio.h>

void shd_dump_module(Module* mod);
void shd_dump_node(const Node* node);

void shd_dump_cfgs(FILE* output, Module* mod);
void shd_dump_loop_trees(FILE* output, Module* mod);

#endif

