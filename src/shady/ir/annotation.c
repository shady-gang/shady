#include "ir_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

static const Node* search_annotations(const Node* node, const char* name, size_t* i) {
    assert(node);
    while (*i < node->annotations.count) {
        const Node* annotation = node->annotations.nodes[*i];
        (*i)++;
        if (strcmp(get_annotation_name(annotation), name) == 0) {
            return annotation;
        }
    }

    return NULL;
}

const Node* shd_lookup_annotation(const Node* node, const char* name) {
    size_t i = 0;
    return search_annotations(node, name, &i);
}

const Node* shd_get_annotation_value(const Node* annotation) {
    assert(annotation);
    if (annotation->tag != AnnotationValue_TAG)
        shd_error("This annotation does not have a single payload");
    return annotation->payload.annotation_value.value;
}

Nodes shd_get_annotation_values(const Node* annotation) {
    assert(annotation);
    if (annotation->tag != AnnotationValues_TAG)
        shd_error("This annotation does not have multiple payloads");
    return annotation->payload.annotation_values.values;
}

/// Gets the string literal attached to an annotation, if present.
const char* shd_get_annotation_string_payload(const Node* annotation) {
    const Node* payload = shd_get_annotation_value(annotation);
    if (!payload) return NULL;
    if (payload->tag != StringLiteral_TAG)
        shd_error("Wrong annotation payload tag, expected a string literal")
    return payload->payload.string_lit.string;
}

bool shd_lookup_annotation_with_string_payload(const Node* node, const char* annotation_name, const char* expected_payload) {
    size_t i = 0;
    while (true) {
        const Node* next = search_annotations(node, annotation_name, &i);
        if (!next) return false;
        if (strcmp(shd_get_annotation_string_payload(next), expected_payload) == 0)
            return true;
    }
}

void shd_add_annotation(const Node* n, const Node* annotation) {
    Node* node = (Node*) n;
    node->annotations = shd_nodes_append(n->arena, node->annotations, annotation);
}

void shd_add_annotation_named(const Node* n, String name) {
    shd_add_annotation(n, annotation_helper(n->arena, name));
}

void shd_remove_annotation_by_name(const Node* n, String name) {
    Node* node = (Node*) n;
    node->annotations = shd_filter_out_annotation(node->arena, node->annotations, name);
}

Nodes shd_filter_out_annotation(IrArena* arena, Nodes annotations, const char* name) {
    LARRAY(const Node*, new_annotations, annotations.count);
    size_t new_count = 0;
    for (size_t i = 0; i < annotations.count; i++) {
        if (strcmp(get_annotation_name(annotations.nodes[i]), name) != 0) {
            new_annotations[new_count++] = annotations.nodes[i];
        }
    }
    return shd_nodes(arena, new_count, new_annotations);
}
