#include "l2s_private.h"

#include "dict.h"
#include "log.h"

#include <stdlib.h>

ParsedAnnotation* find_annotation(Parser* p, const Node* n) {
    return shd_dict_find_value(const Node*, ParsedAnnotation, p->annotations, n);
}

void add_annotation(Parser* p, const Node* n, ParsedAnnotation a) {
    ParsedAnnotation* found = shd_dict_find_value(const Node*, ParsedAnnotation, p->annotations, n);
    if (found) {
        ParsedAnnotation* data = shd_arena_alloc(p->annotations_arena, sizeof(a));
        *data = a;
        while (found->next)
            found = found->next;
        found->next = data;
    } else {
        shd_dict_insert(const Node*, ParsedAnnotation, p->annotations, n, a);
    }
}

static const Node* assert_and_strip_fn_addr(const Node* fn) {
    assert(fn->tag == FnAddr_TAG);
    fn = fn->payload.fn_addr.fn;
    assert(fn->tag == Function_TAG);
    return fn;
}

static const Node* look_past_stuff(const Node* thing) {
    if (thing->tag == Constant_TAG) {
        const Node* instr = thing->payload.constant.value;
        assert(instr->tag == PrimOp_TAG);
        thing = instr;
    }
    if (thing->tag == PrimOp_TAG) {
        switch (thing->payload.prim_op.op) {
            case reinterpret_op:
            case convert_op: thing = first(thing->payload.prim_op.operands); break;
            default: assert(false);
        }
    }
    if (thing->tag == PtrCompositeElement_TAG) {
        thing = thing->payload.ptr_composite_element.ptr;
    }
    return thing;
}

static bool is_io_as(AddressSpace as) {
    switch (as) {
        case AsInput:
        case AsUInput:
        case AsOutput:
        case AsUniform:
        case AsUniformConstant: return true;
        default: break;
    }
    return false;
}

void process_llvm_annotations(Parser* p, LLVMValueRef global) {
    IrArena* a = get_module_arena(p->dst);
    const Type* t = convert_type(p, LLVMGlobalGetValueType(global));
    assert(t->tag == ArrType_TAG);
    size_t arr_size = get_int_literal_value(*resolve_to_int_literal(t->payload.arr_type.size), false);
    assert(arr_size > 0);
    const Node* value = convert_value(p, LLVMGetInitializer(global));
    assert(value->tag == Composite_TAG && value->payload.composite.contents.count == arr_size);
    for (size_t i = 0; i < arr_size; i++) {
        const Node* entry = value->payload.composite.contents.nodes[i];
        entry = look_past_stuff(entry);
        assert(entry->tag == Composite_TAG);
        const Node* annotation_payload = entry->payload.composite.contents.nodes[1];
        // eliminate dummy reinterpret cast
        annotation_payload = look_past_stuff(annotation_payload);
        if (annotation_payload->tag == RefDecl_TAG) {
            annotation_payload = annotation_payload->payload.ref_decl.decl;
        }
        if (annotation_payload->tag == GlobalVariable_TAG) {
            annotation_payload = annotation_payload->payload.global_variable.init;
        }

        NodeResolveConfig resolve_config = default_node_resolve_config();
        // both of those assumptions are hacky but this front-end is a hacky deal anyways.
        resolve_config.assume_globals_immutability = true;
        resolve_config.allow_incompatible_types = true;
        const char* ostr = get_string_literal(a, chase_ptr_to_source(annotation_payload, resolve_config));
        char* str = calloc(strlen(ostr) + 1, 1);
        memcpy(str, ostr, strlen(ostr) + 1);
        if (strcmp(strtok(str, "::"), "shady") == 0) {
            const Node* target = entry->payload.composite.contents.nodes[0];
            target = resolve_node_to_definition(target, resolve_config);

            char* keyword = strtok(NULL, "::");
            if (strcmp(keyword, "entry_point") == 0) {
                target = assert_and_strip_fn_addr(target);
                add_annotation(p, target,  (ParsedAnnotation) {
                    .payload = annotation_value(a, (AnnotationValue) {
                        .name = "EntryPoint",
                        .value = string_lit_helper(a, strtok(NULL, "::"))
                    })
                });
            } else if (strcmp(keyword, "workgroup_size") == 0) {
                target = assert_and_strip_fn_addr(target);
                add_annotation(p, target,  (ParsedAnnotation) {
                    .payload = annotation_values(a, (AnnotationValues) {
                        .name = "WorkgroupSize",
                        .values = mk_nodes(a, int32_literal(a, strtol(strtok(NULL, "::"), NULL, 10)), int32_literal(a, strtol(strtok(NULL, "::"), NULL, 10)), int32_literal(a, strtol(strtok(NULL, "::"), NULL, 10)))
                    })
                });
            } else if (strcmp(keyword, "builtin") == 0) {
                assert(target->tag == GlobalVariable_TAG);
                add_annotation(p, target, (ParsedAnnotation) {
                    .payload = annotation_value(a, (AnnotationValue) {
                        .name = "Builtin",
                        .value = string_lit_helper(a, strtok(NULL, "::"))
                    })
                });
            } else if (strcmp(keyword, "location") == 0) {
                assert(target->tag == GlobalVariable_TAG);
                add_annotation(p, target, (ParsedAnnotation) {
                    .payload = annotation_value(a, (AnnotationValue) {
                        .name = "Location",
                        .value = int32_literal(a, strtol(strtok(NULL, "::"), NULL, 10))
                    })
                });
            } else if (strcmp(keyword, "descriptor_set") == 0) {
                assert(target->tag == GlobalVariable_TAG);
                add_annotation(p, target, (ParsedAnnotation) {
                    .payload = annotation_value(a, (AnnotationValue) {
                        .name = "DescriptorSet",
                        .value = int32_literal(a, strtol(strtok(NULL, "::"), NULL, 10))
                    })
                });
            } else if (strcmp(keyword, "descriptor_binding") == 0) {
                assert(target->tag == GlobalVariable_TAG);
                add_annotation(p, target, (ParsedAnnotation) {
                    .payload = annotation_value(a, (AnnotationValue) {
                        .name = "DescriptorBinding",
                        .value = int32_literal(a, strtol(strtok(NULL, "::"), NULL, 10))
                    })
                });
            } else if (strcmp(keyword, "extern") == 0) {
                assert(target->tag == GlobalVariable_TAG);
                AddressSpace as = convert_llvm_address_space(strtol(strtok(NULL, "::"), NULL, 10));
                if (is_io_as(as))
                    ((Node*) target)->payload.global_variable.init = NULL;
                add_annotation(p, target, (ParsedAnnotation) {
                    .payload = annotation_value(a, (AnnotationValue) {
                        .name = "AddressSpace",
                        .value = int32_literal(a, as)
                    })
                });
            } else {
                shd_error_print("Unrecognised shady annotation '%s'\n", keyword);
                shd_error_die();
            }
        } else {
            shd_warn_print("Ignoring annotation '%s'\n", ostr);
        }
        free(str);
        //dump_node(annotation_payload);
    }
}
