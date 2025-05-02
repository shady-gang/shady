#include "shady/dict.h"

#include "dict.h"

KeyHash shd_hash_node(Node** pnode);
bool shd_compare_node(Node** pa, Node** pb);

NodeSet shd_new_node_set(void) { return shd_new_set(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node); }
bool shd_node_set_find(NodeSet set, const Node* key) { return shd_dict_find_key_fast(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node, set, key); }
bool shd_node_set_insert(NodeSet set, const Node* key) { return shd_set_insert_fast(const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node, set, key); }
bool shd_node_set_iter(Node2Node set, size_t* index, const Node** key) { return shd_dict_iter(set, index, key, NULL); }
void shd_destroy_node_set(NodeSet set) { shd_destroy_dict(set); }

Node2Node shd_new_node2node(void) { return shd_new_dict(const Node*, const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node); }
const Node* shd_node2node_find(Node2Node set, const Node* key) { const Node** found = shd_dict_find_value_fast(const Node*, const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node, set, key); return found ? *found : NULL; }
bool shd_node2node_insert(Node2Node set, const Node* key, const Node* value) { return shd_dict_insert_fast(const Node*, const Node*, (HashFn) shd_hash_node, (CmpFn) shd_compare_node, set, key, value); }
bool shd_node2node_iter(Node2Node set, size_t* index, const Node** key, const Node** value) { return shd_dict_iter(set, index, key, value); }
void shd_destroy_node2node(Node2Node set) { shd_destroy_dict(set); }
