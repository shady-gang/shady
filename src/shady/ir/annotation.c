#include "ir_private.h"

#include "log.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

static const Node* search_annotations(const Node* decl, const char* name, size_t* i) {
    assert(decl);
    const Nodes annotations = get_declaration_annotations(decl);
    while (*i < annotations.count) {
        const Node* annotation = annotations.nodes[*i];
        (*i)++;
        if (strcmp(get_annotation_name(annotation), name) == 0) {
            return annotation;
        }
    }

    return NULL;
}

const Node* shd_lookup_annotation(const Node* decl, const char* name) {
    size_t i = 0;
    return search_annotations(decl, name, &i);
}

const Node* shd_lookup_annotation_list(Nodes annotations, const char* name) {
    for (size_t i = 0; i < annotations.count; i++) {
        if (strcmp(get_annotation_name(annotations.nodes[i]), name) == 0) {
            return annotations.nodes[i];
        }
    }
    return NULL;
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

bool shd_lookup_annotation_with_string_payload(const Node* decl, const char* annotation_name, const char* expected_payload) {
    size_t i = 0;
    while (true) {
        const Node* next = search_annotations(decl, annotation_name, &i);
        if (!next) return false;
        if (strcmp(shd_get_annotation_string_payload(next), expected_payload) == 0)
            return true;
    }
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

ExecutionModel shd_execution_model_from_string(const char* string) {
#define EM(n, _) if (strcmp(string, #n) == 0) return Em##n;
    EXECUTION_MODELS(EM)
#undef EM
    return EmNone;
}
