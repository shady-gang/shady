#ifndef SHADY_IR_DICT_H
#define SHADY_IR_DICT_H

#include "shady/ir/base.h"

typedef struct Dict* NodeSet;
NodeSet shd_new_node_set(void);
bool shd_node_set_find(NodeSet, const Node*);
bool shd_node_set_insert(NodeSet, const Node*);
bool shd_node_set_iter(NodeSet, size_t* index, const Node** key);
void shd_destroy_node_set(NodeSet);

typedef struct Dict* Node2Node;
Node2Node shd_new_node2node(void);
const Node* shd_node2node_find(Node2Node, const Node*);
bool shd_node2node_insert(Node2Node, const Node*, const Node*);
bool shd_node2node_iter(Node2Node, size_t* index, const Node** key, const Node** value);
void shd_destroy_node2node(Node2Node);

#endif
