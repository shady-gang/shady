#ifndef SHD_BE_DUMP_H
#define SHD_BE_DUMP_H

void shd_dump_module(Module* mod);
void shd_dump_node(const Node* node);

void dump_cfgs(FILE* output, Module* mod);
void dump_loop_trees(FILE* output, Module* mod);

#endif

