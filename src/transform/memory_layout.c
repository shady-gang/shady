#include "memory_layout.h"

#include "../log.h"

TypeMemLayout get_mem_layout(const Type* type) {
    switch (type->tag) {
        case FnType_TAG:  error("Functions have an opaque memory representation");
        case PtrType_TAG: error("TODO");
        case Int_TAG:     return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
        };
        case Float_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
        };
        case Bool_TAG:   return (TypeMemLayout) {
            .type = type,
            .size_in_bytes = 4,
        };
        case QualifiedType_TAG: return get_mem_layout(type->payload.qualified_type.type);
        case RecordType_TAG: error("TODO");
        default: error("not a known type");
    }
}
