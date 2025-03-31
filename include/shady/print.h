#ifndef SHADY_PRINT
#define SHADY_PRINT

#include "shady/ir/base.h"
#include "shady/be/dump.h"

#include "printer.h"

typedef struct NodePrintConfig_ {
    bool print_internal;
    bool print_generated;
    bool print_ptrs;
    bool color;
    bool reparseable;
    bool in_cfg;
    int max_depth;
    bool only_immediate;
    bool scheduled;
    bool function_bodies;
} NodePrintConfig;

NodePrintConfig* shd_default_node_print_config(void);

void shd_print_module_into_str(Module* mod, char** str_ptr, size_t* size);
void shd_print_node_into_str(const Node* node, char** str_ptr, size_t* size);

void shd_print_module(Printer* printer, NodePrintConfig config, Module* mod);
void shd_print_node(Printer* printer, NodePrintConfig config, const Node* node);

#endif