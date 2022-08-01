#include "shady/ir.h"
#include "log.h"

#include <assert.h>
#include <string.h>

const Node* lookup_annotation(const Node* decl, const char* name) {
    assert(decl);
    const Nodes* annotations = NULL;
    switch (decl->tag) {
        case Function_TAG: annotations = &decl->payload.fn.annotations; break;
        case GlobalVariable_TAG: annotations = &decl->payload.global_variable.annotations; break;
        case Constant_TAG: annotations = &decl->payload.constant.annotations; break;
        default: error("Not a declaration")
    }

    for (size_t i = 0; i < annotations->count; i++) {
        const Node* annotation = annotations->nodes[i];
        assert(annotation->tag == Annotation_TAG);
        if (strcmp(annotation->payload.annotation.name, name) == 0)
            return annotation;
    }

    return NULL;
}

const Node* extract_annotation_payload(const Node* annotation) {
    if (!annotation) return NULL;
    if (annotation->payload.annotation.payload_type != AnPayloadValue)
        error("This annotation does not have a single payload");
    return annotation->payload.annotation.value;
}

const Nodes* extract_annotation_payloads(const Node* annotation) {
    if (!annotation) return NULL;
    if (annotation->payload.annotation.payload_type != AnPayloadValues)
        error("This annotation does not have multiple payloads");
    return &annotation->payload.annotation.values;
}

/// Gets the string literal attached to an annotation, if present.
const char*  extract_annotation_string_payload(const Node* annotation) {
    const Node* payload = extract_annotation_payload(annotation);
    if (!payload) return NULL;
    if (payload->tag != StringLiteral_TAG)
        error("Wrong annotation payload tag, expected a string literal")
    return payload->payload.string_lit.string;
}
