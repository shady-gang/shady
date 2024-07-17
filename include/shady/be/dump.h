#ifndef SHD_BE_DUMP_H
#define SHD_BE_DUMP_H

void print_module_into_str(Module* mod, char** str_ptr, size_t* size);
void print_node_into_str(const Node* node, char** str_ptr, size_t* size);

void dump_module(Module*);
void dump_node(const Node* node);

void dump_cfgs(FILE* output, Module* mod);
void dump_loop_trees(FILE* output, Module* mod);

#endif

