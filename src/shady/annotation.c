#include "ir_private.h"
#include "log.h"
#include "portability.h"

#include <assert.h>
#include <string.h>

bool is_annotation(const Node* node) {
    switch (node->tag) {
        case Annotation_TAG:
        case AnnotationValue_TAG:
        case AnnotationValues_TAG:
        case AnnotationCompound_TAG: return true;
        default: return false;
    }
}

String get_annotation_name(const Node* node) {
    assert(is_annotation(node));
    switch (node->tag) {
        case Annotation_TAG:      return node->payload.annotation.name;
        case AnnotationValue_TAG: return node->payload.annotation_value.name;
        case AnnotationValues_TAG: return node->payload.annotation_values.name;
        case AnnotationCompound_TAG: return node->payload.annotations_compound.name;
        default: return false;
    }
}

static const Node* search_annotations(const Node* decl, const char* name, size_t* i) {
    assert(decl);
    const Nodes* annotations = NULL;
    switch (decl->tag) {
        case Function_TAG: annotations = &decl->payload.fun.annotations; break;
        case GlobalVariable_TAG: annotations = &decl->payload.global_variable.annotations; break;
        case Constant_TAG: annotations = &decl->payload.constant.annotations; break;
        case NominalType_TAG: annotations = &decl->payload.nom_type.annotations; break;
        default: error("Not a declaration")
    }

    while (*i < annotations->count) {
        const Node* annotation = annotations->nodes[*i];
        (*i)++;
        if (strcmp(get_annotation_name(annotation), name) == 0) {
            return annotation;
        }
    }

    return NULL;
}

const Node* lookup_annotation(const Node* decl, const char* name) {
    size_t i = 0;
    return search_annotations(decl, name, &i);
}

const Node* lookup_annotation_list(Nodes annotations, const char* name) {
    for (size_t i = 0; i < annotations.count; i++) {
        if (strcmp(get_annotation_name(annotations.nodes[i]), name) == 0) {
            return annotations.nodes[i];
        }
    }
    return NULL;
}

const Node* get_annotation_value(const Node* annotation) {
    assert(annotation);
    if (annotation->tag != AnnotationValue_TAG)
        error("This annotation does not have a single payload");
    return annotation->payload.annotation_value.value;
}

Nodes get_annotation_values(const Node* annotation) {
    assert(annotation);
    if (annotation->tag != AnnotationValues_TAG)
        error("This annotation does not have multiple payloads");
    return annotation->payload.annotation_values.values;
}

/// Gets the string literal attached to an annotation, if present.
const char* get_annotation_string_payload(const Node* annotation) {
    const Node* payload = get_annotation_value(annotation);
    if (!payload) return NULL;
    if (payload->tag != StringLiteral_TAG)
        error("Wrong annotation payload tag, expected a string literal")
    return payload->payload.string_lit.string;
}

bool lookup_annotation_with_string_payload(const Node* decl, const char* annotation_name, const char* expected_payload) {
    size_t i = 0;
    while (true) {
        const Node* next = search_annotations(decl, annotation_name, &i);
        if (!next) return false;
        if (strcmp(get_annotation_string_payload(next), expected_payload) == 0)
            return true;
    }
}

Nodes filter_out_annotation(IrArena* arena, Nodes annotations, const char* name) {
    LARRAY(const Node*, new_annotations, annotations.count);
    size_t new_count = 0;
    for (size_t i = 0; i < annotations.count; i++) {
        if (strcmp(get_annotation_name(annotations.nodes[i]), name) != 0) {
            new_annotations[new_count++] = annotations.nodes[i];
        }
    }
    return nodes(arena, new_count, new_annotations);
}

ExecutionModel execution_model_from_string(const char* string) {
#define EM(n, _) if (strcmp(string, #n) == 0) return Em##n;
    EXECUTION_MODELS(EM)
#undef EM
    return EmNone;
}
