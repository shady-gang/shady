#include "l2s_private.h"

#include "dict.h"
#include "log.h"

#include <stdlib.h>

ParsedAnnotationContents* find_annotation(Parser* p, const Node* n, AnnotationType t) {
    ParsedAnnotationContents* a = find_value_dict(const Node*, ParsedAnnotationContents, p->annotations, n);
    if (!a)
        return NULL;
    else if (t == NoneAnnot || a->type == t)
        return a;
    else
        return next_annotation(a, t);
}

ParsedAnnotationContents* next_annotation(ParsedAnnotationContents* a, AnnotationType t) {
    do {
        a = a->next;
        if (a && (t == NoneAnnot || a->type == t))
            return a;
    } while (a);
    return a;
}

void add_annotation(Parser* p, const Node* n, ParsedAnnotationContents a) {
    ParsedAnnotationContents* found = find_value_dict(const Node*, ParsedAnnotationContents, p->annotations, n);
    if (found) {
        ParsedAnnotationContents* data = arena_alloc(p->annotations_arena, sizeof(a));
        *data = a;
        while (found->next)
            found = found->next;
        found->next = data;
    } else {
        insert_dict(const Node*, ParsedAnnotationContents, p->annotations, n, a);
    }
}

void process_llvm_annotations(Parser* p, LLVMValueRef global) {
    IrArena* a = get_module_arena(p->dst);
    const Type* t = convert_type(p, LLVMGlobalGetValueType(global));
    assert(t->tag == ArrType_TAG);
    size_t arr_size = get_int_literal_value(t->payload.arr_type.size, false);
    assert(arr_size > 0);
    const Node* value = convert_value(p, LLVMGetInitializer(global));
    assert(value->tag == Composite_TAG && value->payload.composite.contents.count == arr_size);
    for (size_t i = 0; i < arr_size; i++) {
        const Node* entry = value->payload.composite.contents.nodes[i];
        assert(entry->tag == Composite_TAG);
        const Node* annotation_payload = entry->payload.composite.contents.nodes[1];
        // eliminate dummy reinterpret cast
        if (annotation_payload->tag == AntiQuote_TAG) {
            const Node* instr = annotation_payload->payload.anti_quote.instruction;
            assert(instr->tag == PrimOp_TAG);
            switch (instr->payload.prim_op.op) {
                case reinterpret_op:
                case lea_op: annotation_payload = first(annotation_payload->payload.anti_quote.instruction->payload.prim_op.operands); break;
                default: assert(false);
            }
        }
        if (annotation_payload->tag == RefDecl_TAG) {
            annotation_payload = annotation_payload->payload.ref_decl.decl;
        }
        if (annotation_payload->tag == GlobalVariable_TAG) {
            annotation_payload = annotation_payload->payload.global_variable.init;
        }
        const char* ostr = get_string_literal(a, annotation_payload);
        char* str = calloc(strlen(ostr) + 1, 1);
        memcpy(str, ostr, strlen(ostr) + 1);
        if (strcmp(strtok(str, "::"), "shady") == 0) {
            const Node* target = entry->payload.composite.contents.nodes[0];
            if (target->tag == AntiQuote_TAG) {
                assert(target->payload.anti_quote.instruction->tag == PrimOp_TAG);
                assert(target->payload.anti_quote.instruction->payload.prim_op.op == reinterpret_op);
                target = first(target->payload.anti_quote.instruction->payload.prim_op.operands);
            }
            if (target->tag == RefDecl_TAG) {
                target = target->payload.ref_decl.decl;
            }

            char* keyword = strtok(NULL, "::");
            if (strcmp(keyword, "entry_point") == 0) {
                assert(target->tag == Function_TAG);
                add_annotation(p, target, (ParsedAnnotationContents) {
                        .type = EntryPointAnnot,
                        .payload.entry_point_type = strtok(NULL, "::")
                });
            } else if (strcmp(keyword, "builtin") == 0) {
                assert(target->tag == GlobalVariable_TAG);
                add_annotation(p, target, (ParsedAnnotationContents) {
                        .type = BuiltinAnnot,
                        .payload.builtin_name = strtok(NULL, "::")
                });
            } else {
                error_print("Unrecognised shady annotation '%s'\n", keyword);
                error_die();
            }
        } else {
            warn_print("Ignoring annotation '%s'\n", ostr);
        }
        free(str);
        //dump_node(annotation_payload);
    }
}