#include "shady/ir/execution_model.h"
#include "shady/ir/grammar.h"
#include "shady/ir/debug.h"
#include "shady/ir/annotation.h"
#include "shady/ir/int.h"

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

bool shd_get_workgroup_size_for_entry_point(const Node* decl, uint32_t* out) {
    const Node* old_wg_size_annotation = shd_lookup_annotation(decl, "WorkgroupSize");
    if (old_wg_size_annotation && old_wg_size_annotation->tag == AnnotationValues_TAG && shd_get_annotation_values(old_wg_size_annotation).count == 3) {
        Nodes wg_size_nodes = shd_get_annotation_values(old_wg_size_annotation);
        out[0] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[0]), false);
        out[1] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[1]), false);
        out[2] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[2]), false);
        return true;
    }
    return false;
}
