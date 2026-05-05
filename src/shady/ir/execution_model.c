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

ExecutionModelInfo shd_get_execution_model_info_from_entry_point(const Node* fn) {
    ExecutionModelInfo exec_info = {
        .execution_model = shd_execution_model_from_entry_point(fn)
    };
    switch (exec_info.execution_model) {
        case ShdExecutionModelNone:
            break;
        case ShdExecutionModelCompute: {
            const Node* wg_size_annotation = shd_lookup_annotation(fn, "WorkgroupSize");
            if (wg_size_annotation && wg_size_annotation->tag == AnnotationValues_TAG && shd_get_annotation_values(wg_size_annotation).count == 3) {
                Nodes wg_size_nodes = shd_get_annotation_values(wg_size_annotation);
                exec_info.grid_based.workgroup_size[0] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[0]), false);
                exec_info.grid_based.workgroup_size[1] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[1]), false);
                exec_info.grid_based.workgroup_size[2] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[2]), false);
            } else {
                shd_warn_print("Missing workgroup size from node ");
                shd_log_node(WARN, fn);
                shd_warn_print(".\n");
            }
            break;
        }
        case ShdExecutionModelMesh:{
            const Node* wg_size_annotation = shd_lookup_annotation(fn, "WorkgroupSize");
            const Node* num_vertices_annotation = shd_lookup_annotation(fn, "NumVertices");
            const Node* num_primitives_annotation = shd_lookup_annotation(fn, "NumPrimitives");

            if (wg_size_annotation && wg_size_annotation->tag == AnnotationValues_TAG && shd_get_annotation_values(wg_size_annotation).count == 3) {
                Nodes wg_size_nodes = shd_get_annotation_values(wg_size_annotation);
                exec_info.grid_based.workgroup_size[0] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[0]), false);
                exec_info.grid_based.workgroup_size[1] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[1]), false);
                exec_info.grid_based.workgroup_size[2] = shd_get_int_literal_value(*shd_resolve_to_int_literal(wg_size_nodes.nodes[2]), false);
            } else {
                shd_warn_print("Missing workgroup size from node ");
                shd_log_node(WARN, fn);
                shd_warn_print(".\n");
            }

            if (num_vertices_annotation && num_vertices_annotation->tag == AnnotationValues_TAG && shd_get_annotation_values(num_vertices_annotation).count == 1) {
                Nodes num_vertices_nodes = shd_get_annotation_values(num_vertices_annotation);
                exec_info.num_vertices = shd_get_int_literal_value(*shd_resolve_to_int_literal(*num_vertices_nodes.nodes), false);
            } else {
                shd_warn_print("Missing workgroup size from node ");
                shd_log_node(WARN, fn);
                shd_warn_print(".\n");
            }

            if (num_primitives_annotation && num_primitives_annotation->tag == AnnotationValues_TAG && shd_get_annotation_values(num_primitives_annotation).count == 1) {
                Nodes num_primitives_nodes = shd_get_annotation_values(num_primitives_annotation);
                exec_info.num_primitives = shd_get_int_literal_value(*shd_resolve_to_int_literal(*num_primitives_nodes.nodes), false);
            } else {
                shd_warn_print("Missing workgroup size from node ");
                shd_log_node(WARN, fn);
                shd_warn_print(".\n");
            }
            break;
        }
        case ShdExecutionModelFragment:
            break;
        case ShdExecutionModelVertex:
            break;
        case ShdExecutionModelRayGeneration:
            break;
        case ShdExecutionModelCallable:
            break;
        default:
            assert(false);
    }
    return exec_info;
}

bool shd_get_workgroup_size(const ExecutionModelInfo* info, uint32_t* out) {
    if (shd_is_execution_model_workgroup_based(info->execution_model)) {
        out[0] = info->grid_based.workgroup_size[0];
        out[1] = info->grid_based.workgroup_size[1];
        out[2] = info->grid_based.workgroup_size[2];
        return true;
    }
    return false;
}

bool shd_get_num_vertices(const ExecutionModelInfo* info, uint32_t* out) {
    if (shd_is_execution_model_workgroup_based(info->execution_model)) {
        out[0] = info->num_vertices;
        return true;
    }
    return false;
}

bool shd_get_num_primitives(const ExecutionModelInfo* info, uint32_t* out) {
    if (shd_is_execution_model_workgroup_based(info->execution_model)) {
        out[0] = info->num_primitives;
        return true;
    }
    return false;
}

inline static size_t div_roundup(size_t a, size_t b) {
    if (a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

bool shd_get_num_subgroups_per_workgroups(const ExecutionModelInfo* info, uint32_t subgroup_size, uint32_t* out) {
    uint32_t wg_size[3];
    if (!shd_get_workgroup_size(info, wg_size))
        return false;
    assert(wg_size[0] * wg_size[1] * wg_size[2] > 0);
    uint32_t subgroups_per_wg = div_roundup(wg_size[0] * wg_size[1] * wg_size[2], subgroup_size);
    assert(subgroups_per_wg != 0);
    *out = subgroups_per_wg;
    return true;
}
