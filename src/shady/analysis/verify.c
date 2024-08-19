#include "verify.h"
#include "free_frontier.h"
#include "cfg.h"

#include "log.h"
#include "dict.h"
#include "list.h"

#include "../visit.h"
#include "../ir_private.h"
#include "../type.h"

#include <assert.h>

typedef struct {
    Visitor visitor;
    const IrArena* arena;
    struct Dict* once;
} ArenaVerifyVisitor;

static void visit_verify_same_arena(ArenaVerifyVisitor* visitor, const Node* node) {
    assert(visitor->arena == node->arena);
    if (find_key_dict(const Node*, visitor->once, node))
        return;
    insert_set_get_result(const Node*, visitor->once, node);
    visit_node_operands(&visitor->visitor, 0, node);
}

KeyHash hash_node(const Node**);
bool compare_node(const Node**, const Node**);

static void verify_same_arena(Module* mod) {
    const IrArena* arena = get_module_arena(mod);
    ArenaVerifyVisitor visitor = {
        .visitor = {
            .visit_node_fn = (VisitNodeFn) visit_verify_same_arena,
        },
        .arena = arena,
        .once = new_set(const Node*, (HashFn) hash_node, (CmpFn) compare_node)
    };
    visit_module(&visitor.visitor, mod);
    destroy_dict(visitor.once);
}

static void verify_scoping(const CompilerConfig* config, Module* mod) {
    struct List* cfgs = build_cfgs(mod);
    for (size_t i = 0; i < entries_count_list(cfgs); i++) {
        CFG* cfg = read_list(CFG*, cfgs)[i];
        Scheduler* scheduler = new_scheduler(cfg);
        struct Dict* set = free_frontier(scheduler, cfg, cfg->entry->node);
        if (entries_count_dict(set) > 0) {
            log_string(ERROR, "Leaking variables in ");
            log_node(ERROR, cfg->entry->node);
            log_string(ERROR, ":\n");

            size_t j = 0;
            const Node* leaking;
            while (dict_iter(set, &j, &leaking, NULL)) {
                log_node(ERROR, leaking);
                error_print("\n");
            }

            log_string(ERROR, "Problematic module:\n");
            log_module(ERROR, config, mod);
            error_die();
        }
        destroy_dict(set);
        destroy_scheduler(scheduler);
        destroy_cfg(cfg);
    }
    destroy_list(cfgs);
}

static void verify_nominal_node(const Node* fn, const Node* n) {
    switch (n->tag) {
        case Function_TAG: {
            assert(!fn && "functions cannot be part of a CFG, except as the entry");
            break;
        }
        case BasicBlock_TAG: {
            assert(is_subtype(noret_type(n->arena), n->payload.basic_block.body->type));
            break;
        }
        case NominalType_TAG: {
            assert(is_type(n->payload.nom_type.body));
            break;
        }
        case Constant_TAG: {
            if (n->payload.constant.value) {
                const Type* t = n->payload.constant.value->type;
                bool u = deconstruct_qualified_type(&t);
                assert(u);
                assert(is_subtype(n->payload.constant.type_hint, t));
            }
            break;
        }
        case GlobalVariable_TAG: {
            if (n->payload.global_variable.init) {
                const Type* t = n->payload.global_variable.init->type;
                bool u = deconstruct_qualified_type(&t);
                assert(u);
                assert(is_subtype(n->payload.global_variable.type, t));
            }
            break;
        }
        default: break;
    }
}

typedef struct ScheduleContext_ {
    Visitor visitor;
    struct Dict* bound;
    struct ScheduleContext_* parent;
    CompilerConfig* config;
    Module* mod;
} ScheduleContext;

static void verify_schedule_visitor(ScheduleContext* ctx, const Node* node) {
    if (is_instruction(node)) {
        ScheduleContext* search = ctx;
        while (search) {
            if (find_key_dict(const Node*, search->bound, node))
                break;
            search = search->parent;
        }
        if (!search) {
            log_string(ERROR, "Scheduling problem: ");
            log_node(ERROR, node);
            log_string(ERROR, "was encountered before we saw it be bound by a let!\n");
            log_string(ERROR, "Problematic module:\n");
            log_module(ERROR, ctx->config, ctx->mod);
            error_die();
        }
    }
    visit_node_operands(&ctx->visitor, NcTerminator | NcDeclaration, node);
}

static void verify_bodies(const CompilerConfig* config, Module* mod) {
    struct List* cfgs = build_cfgs(mod);
    for (size_t i = 0; i < entries_count_list(cfgs); i++) {
        CFG* cfg = read_list(CFG*, cfgs)[i];

        for (size_t j = 0; j < cfg->size; j++) {
            CFNode* n = cfg->rpo[j];
            if (n->node->tag == BasicBlock_TAG) {
                verify_nominal_node(cfg->entry->node, n->node);
            }
        }

        destroy_cfg(cfg);
    }
    destroy_list(cfgs);

    Nodes decls = get_module_declarations(mod);
    for (size_t i = 0; i < decls.count; i++) {
        const Node* decl = decls.nodes[i];
        verify_nominal_node(NULL, decl);
    }
}

void verify_module(const CompilerConfig* config, Module* mod) {
    verify_same_arena(mod);
    // before we normalize the IR, scopes are broken because decls appear where they should not
    // TODO add a normalized flag to the IR and check grammar is adhered to strictly
    if (get_module_arena(mod)->config.check_types) {
        verify_scoping(config, mod);
        verify_bodies(config, mod);
    }
}
