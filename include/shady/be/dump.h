#ifndef SHD_BE_DUMP_H
#define SHD_BE_DUMP_H

void dump_module(Module*);
void dump_node(const Node* node);

void dump_cfgs(FILE* output, Module* mod);
void dump_loop_trees(FILE* output, Module* mod);

#endif

