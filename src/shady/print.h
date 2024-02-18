#ifndef SHADY_PRINT
#define SHADY_PRINT

#include "shady/ir.h"
#include "printer.h"
#include <stdbool.h>

typedef struct {
    bool skip_builtin;
    bool skip_internal;
    bool skip_generated;
    bool print_ptrs;
    bool color;
    bool reparseable;
} PrintConfig;

void print_module(Printer* printer, Module* mod, PrintConfig config);
void print_node(Printer* printer, const Node* node, PrintConfig config);
void print_node_operand(Printer* printer, const Node* node, String op_name, NodeClass op_class, const Node* op, PrintConfig config);
void print_node_operand_list(Printer* printer, const Node* node, String op_name, NodeClass op_class, Nodes ops, PrintConfig config);

void print_node_generated(Printer* printer, const Node* node, PrintConfig config);

void print_module_into_str(Module* mod, char** str_ptr, size_t* size);
void print_node_into_str(const Node* node, char** str_ptr, size_t* size);

void dump_module(Module*);
void dump_node(const Node* node);

#endif