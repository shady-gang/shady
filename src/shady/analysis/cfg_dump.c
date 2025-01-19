#include "cfg.h"

#include "ir_private.h"
#include "shady/print.h"

#include "list.h"
#include "dict.h"
#include "util.h"
#include "printer.h"

#include <string.h>
#include <assert.h>

static int extra_uniqueness = 0;

static void print_node_helper(Printer* p, const Node* n) {
    Growy* tmp_g = shd_new_growy();
    Printer* tmp_p = shd_new_printer_from_growy(tmp_g);

    NodePrintConfig config = {
        .color = false,
        .in_cfg = true,
    };

    shd_print_node(tmp_p, config, n);

    String label = shd_printer_growy_unwrap(tmp_p);
    char* escaped_label = calloc(strlen(label) * 2, 1);
    shd_unapply_escape_codes(label, strlen(label), escaped_label);

    shd_print(p, "%s", escaped_label);
    free(escaped_label);
    free((void*)label);
}

static const Nodes* find_scope_info(const Node* abs) {
    assert(is_abstraction(abs));
    const Node* terminator = get_abstraction_body(abs);
    const Node* mem = get_terminator_mem(terminator);
    Nodes* info = NULL;
    while (mem) {
        if (mem->tag == ExtInstr_TAG && strcmp(mem->payload.ext_instr.set, "shady.scope") == 0) {
            if (!info || info->count > mem->payload.ext_instr.operands.count)
                info = &mem->payload.ext_instr.operands;
        }
        mem = shd_get_parent_mem(mem);
    }
    return info;
}

static void dump_cf_node(FILE* output, const CFNode* n) {
    const Node* bb = n->node;
    const Node* body = get_abstraction_body(bb);
    if (!body)
        return;

    String color = "black";
    switch (body->tag) {
        case If_TAG: color = "blue"; break;
        case Loop_TAG: color = "red"; break;
        case Control_TAG: color = "orange"; break;
        case Return_TAG: color = "teal"; break;
        case Unreachable_TAG: color = "teal"; break;
        default: break;
    }

    Growy* g = shd_new_growy();
    Printer* p = shd_new_printer_from_growy(g);

    String abs_name = shd_get_node_name_safe(bb);

    Nodes params = get_abstraction_params(bb);
    shd_print(p, "%%%d %s", bb->id, abs_name);
    shd_print(p, "(");
    for (size_t i = 0; i < params.count; i++) {
        const Node* param = params.nodes[i];
        shd_print(p, "%s: %s", shd_get_node_name_safe(param), shd_get_type_name(bb->arena, param->type));
    }
    shd_print(p, "): \n");

    if (getenv("SHADY_CFG_SCOPE_ONLY")) {
        const Nodes* scope = find_scope_info(bb);
        if (scope) {
            for (size_t i = 0; i < scope->count; i++) {
                shd_print(p, "%d", scope->nodes[i]->id);
                if (i + 1 < scope->count)
                    shd_print(p, ", ");
            }
        }
    } else {
        print_node_helper(p, body);
        shd_print(p, "\\l");
    }
    shd_print(p, "rpo: %d, idom: %s, sdom: %s", n->rpo_index, n->idom ? shd_get_node_name_safe(n->idom->node) : "null", n->structured_idom ? shd_get_node_name_safe(n->structured_idom->node) : "null");

    String label = shd_printer_growy_unwrap(p);
    fprintf(output, "bb_%zu [nojustify=true, label=\"%s\", color=\"%s\", shape=box];\n", (size_t) n, label, color);
    free((void*) label);

    //for (size_t i = 0; i < entries_count_list(n->dominates); i++) {
    //    CFNode* d = read_list(CFNode*, n->dominates)[i];
    //    if (!find_key_dict(const Node*, n->structurally_dominates, d->node))
    //    dump_cf_node(output, d);
    //}
}

static void dump_cfg(FILE* output, CFG* cfg) {
    extra_uniqueness++;

    const Node* entry = cfg->entry->node;
    fprintf(output, "subgraph cluster_%d {\n", entry->id);
    fprintf(output, "label = \"%s\";\n", shd_get_node_name_safe(entry));
    for (size_t i = 0; i < shd_list_count(cfg->contents); i++) {
        const CFNode* n = shd_read_list(const CFNode*, cfg->contents)[i];
        dump_cf_node(output, n);
    }
    for (size_t i = 0; i < shd_list_count(cfg->contents); i++) {
        const CFNode* bb_node = shd_read_list(const CFNode*, cfg->contents)[i];
        const CFNode* src_node = bb_node;

        for (size_t j = 0; j < shd_list_count(bb_node->succ_edges); j++) {
            CFEdge edge = shd_read_list(CFEdge, bb_node->succ_edges)[j];
            const CFNode* target_node = edge.dst;
            String edge_color = "black";
            String edge_style = "solid";
            switch (edge.type) {
                case StructuredEnterBodyEdge: edge_color = "blue"; break;
                case StructuredLeaveBodyEdge: edge_color = "red"; break;
                case StructuredTailEdge: edge_style = "dashed"; break;
                case StructuredLoopContinue: edge_style = "dotted"; edge_color = "orange"; break;
                default: break;
            }

            fprintf(output, "bb_%zu -> bb_%zu [color=\"%s\", style=\"%s\"];\n", (size_t) (src_node), (size_t) (target_node), edge_color, edge_style);
        }
    }
    fprintf(output, "}\n");
}

void shd_dump_existing_cfg_auto(CFG* cfg) {
    FILE* f = fopen("cfg.dot", "wb");
    fprintf(f, "digraph G {\n");
    dump_cfg(f, cfg);
    fprintf(f, "}\n");
    fclose(f);
}

void shd_dump_cfg_auto(const Node* fn) {
    FILE* f = fopen("cfg.dot", "wb");
    fprintf(f, "digraph G {\n");
    CFG* cfg = build_fn_cfg(fn);
    dump_cfg(f, cfg);
    shd_destroy_cfg(cfg);
    fprintf(f, "}\n");
    fclose(f);
}

void shd_dump_cfgs(FILE* output, Module* mod) {
    if (output == NULL)
        output = stderr;

    fprintf(output, "digraph G {\n");
    struct List* cfgs = shd_build_cfgs(mod, default_forward_cfg_build());
    for (size_t i = 0; i < shd_list_count(cfgs); i++) {
        CFG* cfg = shd_read_list(CFG*, cfgs)[i];
        dump_cfg(output, cfg);
        shd_destroy_cfg(cfg);
    }
    shd_destroy_list(cfgs);
    fprintf(output, "}\n");
}

void shd_dump_cfgs_auto(Module* mod) {
    FILE* f = fopen("cfg.dot", "wb");
    shd_dump_cfgs(f, mod);
    fclose(f);
}

static void dump_domtree_cfnode(Printer* p, CFNode* idom) {
    String name = shd_get_node_name_safe(idom->node);
    if (name)
        shd_print(p, "bb_%zu [label=\"%s\", shape=box];\n", (size_t) idom, name);
    else
        shd_print(p, "bb_%zu [label=\"%%%d\", shape=box];\n", (size_t) idom, idom->node->id);

    for (size_t i = 0; i < shd_list_count(idom->dominates); i++) {
        CFNode* child = shd_read_list(CFNode*, idom->dominates)[i];
        dump_domtree_cfnode(p, child);
        shd_print(p, "bb_%zu -> bb_%zu;\n", (size_t) (idom), (size_t) (child));
    }
}

void shd_dump_domtree_cfg(Printer* p, CFG* s) {
    shd_print(p, "subgraph cluster_%s {\n", shd_get_node_name_safe(s->entry->node));
    dump_domtree_cfnode(p, s->entry);
    shd_print(p, "}\n");
}

void shd_dump_domtree_module(Printer* p, Module* mod) {
    shd_print(p, "digraph G {\n");
    struct List* cfgs = shd_build_cfgs(mod, default_forward_cfg_build());
    for (size_t i = 0; i < shd_list_count(cfgs); i++) {
        CFG* cfg = shd_read_list(CFG*, cfgs)[i];
        shd_dump_domtree_cfg(p, cfg);
        shd_destroy_cfg(cfg);
    }
    shd_destroy_list(cfgs);
    shd_print(p, "}\n");
}

void shd_dump_domtree_auto(Module* mod) {
    FILE* f = fopen("domtree.dot", "wb");
    Printer* p = shd_new_printer_from_file(f);
    shd_dump_domtree_module(p, mod);
    shd_destroy_printer(p);
    fclose(f);
}
