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
    Growy* tmp_g = new_growy();
    Printer* tmp_p = open_growy_as_printer(tmp_g);

    NodePrintConfig config = {
        .color = false,
        .in_cfg = true,
    };

    print_node(tmp_p, config, n);

    String label = printer_growy_unwrap(tmp_p);
    char* escaped_label = calloc(strlen(label) * 2, 1);
    unapply_escape_codes(label, strlen(label), escaped_label);

    print(p, "%s", escaped_label);
    free(escaped_label);
    free((void*)label);
}

static void dump_cf_node(FILE* output, const CFNode* n) {
    const Node* bb = n->node;
    const Node* body = get_abstraction_body(bb);
    if (!body)
        return;

    String color = "black";
    if (is_basic_block(bb))
        color = "blue";

    Growy* g = new_growy();
    Printer* p = open_growy_as_printer(g);

    String abs_name = get_abstraction_name_safe(bb);

    print(p, "%s: \n%d: ", abs_name, bb->id);

    print_node_helper(p, body);
    print(p, "\\l");
    print(p, "rpo: %d, idom: %s, sdom: %s", n->rpo_index, n->idom ? get_abstraction_name_safe(n->idom->node) : "null", n->structured_idom ? get_abstraction_name_safe(n->structured_idom->node) : "null");

    String label = printer_growy_unwrap(p);
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
    fprintf(output, "label = \"%s\";\n", get_abstraction_name_safe(entry));
    for (size_t i = 0; i < entries_count_list(cfg->contents); i++) {
        const CFNode* n = read_list(const CFNode*, cfg->contents)[i];
        dump_cf_node(output, n);
    }
    for (size_t i = 0; i < entries_count_list(cfg->contents); i++) {
        const CFNode* bb_node = read_list(const CFNode*, cfg->contents)[i];
        const CFNode* src_node = bb_node;

        for (size_t j = 0; j < entries_count_list(bb_node->succ_edges); j++) {
            CFEdge edge = read_list(CFEdge, bb_node->succ_edges)[j];
            const CFNode* target_node = edge.dst;
            String edge_color = "black";
            String edge_style = "solid";
            switch (edge.type) {
                case StructuredEnterBodyEdge: edge_color = "blue"; break;
                case StructuredLeaveBodyEdge: edge_color = "red"; break;
                case StructuredTailEdge: edge_style = "dashed"; break;
                case StructuredLoopContinue: edge_style = "dotted"; break;
                default: break;
            }

            fprintf(output, "bb_%zu -> bb_%zu [color=\"%s\", style=\"%s\"];\n", (size_t) (src_node), (size_t) (target_node), edge_color, edge_style);
        }
    }
    fprintf(output, "}\n");
}

void dump_existing_cfg_auto(CFG* cfg) {
    FILE* f = fopen("cfg.dot", "wb");
    fprintf(f, "digraph G {\n");
    dump_cfg(f, cfg);
    fprintf(f, "}\n");
    fclose(f);
}

void dump_cfg_auto(const Node* fn) {
    FILE* f = fopen("cfg.dot", "wb");
    fprintf(f, "digraph G {\n");
    CFG* cfg = build_fn_cfg(fn);
    dump_cfg(f, cfg);
    destroy_cfg(cfg);
    fprintf(f, "}\n");
    fclose(f);
}

void dump_cfgs(FILE* output, Module* mod) {
    if (output == NULL)
        output = stderr;

    fprintf(output, "digraph G {\n");
    struct List* cfgs = build_cfgs(mod, forward_cfg_build(true));
    for (size_t i = 0; i < entries_count_list(cfgs); i++) {
        CFG* cfg = read_list(CFG*, cfgs)[i];
        dump_cfg(output, cfg);
        destroy_cfg(cfg);
    }
    destroy_list(cfgs);
    fprintf(output, "}\n");
}

void dump_cfgs_auto(Module* mod) {
    FILE* f = fopen("cfg.dot", "wb");
    dump_cfgs(f, mod);
    fclose(f);
}

static void dump_domtree_cfnode(Printer* p, CFNode* idom) {
    String name = get_abstraction_name_safe(idom->node);
    if (name)
        print(p, "bb_%zu [label=\"%s\", shape=box];\n", (size_t) idom, name);
    else
        print(p, "bb_%zu [label=\"%%%d\", shape=box];\n", (size_t) idom, idom->node->id);

    for (size_t i = 0; i < entries_count_list(idom->dominates); i++) {
        CFNode* child = read_list(CFNode*, idom->dominates)[i];
        dump_domtree_cfnode(p, child);
        print(p, "bb_%zu -> bb_%zu;\n", (size_t) (idom), (size_t) (child));
    }
}

void dump_domtree_cfg(Printer* p, CFG* s) {
    print(p, "subgraph cluster_%s {\n", get_abstraction_name_safe(s->entry->node));
    dump_domtree_cfnode(p, s->entry);
    print(p, "}\n");
}

void dump_domtree_module(Printer* p, Module* mod) {
    print(p, "digraph G {\n");
    struct List* cfgs = build_cfgs(mod, forward_cfg_build(true));
    for (size_t i = 0; i < entries_count_list(cfgs); i++) {
        CFG* cfg = read_list(CFG*, cfgs)[i];
        dump_domtree_cfg(p, cfg);
        destroy_cfg(cfg);
    }
    destroy_list(cfgs);
    print(p, "}\n");
}

void dump_domtree_auto(Module* mod) {
    FILE* f = fopen("domtree.dot", "wb");
    Printer* p = open_file_as_printer(f);
    dump_domtree_module(p, mod);
    destroy_printer(p);
    fclose(f);
}
