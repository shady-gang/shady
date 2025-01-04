#include "ir_private.h"

#include "log.h"
#include "portability.h"
#include "dict.h"

const char* node_tags[];

const char* shd_get_node_tag_string(NodeTag tag) {
    return node_tags[tag];
}

const bool node_type_is_recursive[];

bool shd_is_node_tag_recursive(NodeTag tag) {
    return node_type_is_recursive[tag];
}

const bool node_type_has_payload[];

KeyHash _shd_hash_node_payload(const Node* node);

KeyHash shd_hash_node(Node** pnode) {
    const Node* node = *pnode;
    KeyHash combined;

    if (shd_is_node_nominal(node)) {
        size_t ptr = (size_t) node;
        uint32_t upper = ptr >> 32;
        uint32_t lower = ptr;
        combined = upper ^ lower;
        goto end;
    }

    KeyHash tag_hash = shd_hash(&node->tag, sizeof(NodeTag));
    KeyHash payload_hash = 0;

    if (node_type_has_payload[node->tag]) {
        payload_hash = _shd_hash_node_payload(node);
    }
    combined = tag_hash ^ payload_hash;

    end:
    return combined;
}

bool _shd_compare_node_payload(const Node*, const Node*);

bool shd_compare_node(Node** pa, Node** pb) {
    if ((*pa)->tag != (*pb)->tag) return false;
    if (shd_is_node_nominal((*pa)))
        return *pa == *pb;

    const Node* a = *pa;
    const Node* b = *pb;

    #undef field
    #define field(w) eq &= memcmp(&a->payload.w, &b->payload.w, sizeof(a->payload.w)) == 0;

    if (node_type_has_payload[a->tag]) {
        return _shd_compare_node_payload(a, b);
    } else return true;
}

#include "node_generated.c"
