#include "shady/ir/execution_model.h"
#include "shady/ir/grammar.h"
#include "shady/ir/debug.h"
#include "shady/ir/annotation.h"

#include "log.h"

#include <string.h>

ShdExecutionModel shd_execution_model_from_string(const char* string) {
#define EM(n) if (strcmp(string, #n) == 0) return ShdExecutionModel##n;
    EXECUTION_MODELS(EM)
#undef EM
    return ShdExecutionModelNone;
}

ShdExecutionModel shd_execution_model_from_entry_point(const Node* decl) {
    String name = shd_get_node_name_safe(decl);
    if (decl->tag != Function_TAG)
        shd_error("Cannot specialize: '%s' is not a function.", name)
    const Node* ep = shd_lookup_annotation(decl, "EntryPoint");
    if (!ep)
        shd_error("%s is not annotated with @EntryPoint", name);
    return shd_execution_model_from_string(shd_get_annotation_string_payload(ep));
}