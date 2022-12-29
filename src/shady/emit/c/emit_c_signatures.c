#include "emit_c.h"

#include "dict.h"
#include "log.h"

#include "../../type.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

void emit_nominal_type_body(Emitter* emitter, String name, const Type* type) {
    assert(type->tag == RecordType_TAG);
    Growy* g = new_growy();
    Printer* p = open_growy_as_printer(g);

    print(p, "\n%s {", name);
    indent(p);
    for (size_t i = 0; i < type->payload.record_type.members.count; i++) {
        String member_identifier;
        if (i >= type->payload.record_type.names.count)
            member_identifier = format_string(emitter->arena, "_%d", i);
        else
            member_identifier = type->payload.record_type.names.strings[i];

        print(p, "\n%s;", emit_type(emitter, type->payload.record_type.members.nodes[i], member_identifier));
    }
    deindent(p);
    print(p, "\n};\n");
    growy_append_bytes(g, 1, (char[]) { '\0' });

    print(emitter->type_decls, growy_data(g));
    destroy_growy(g);
    destroy_printer(p);
}

String emit_type(Emitter* emitter, const Type* type, const char* center) {
    if (center == NULL)
        center = "";

    String emitted = NULL;
    CType* found = lookup_existing_type(emitter, type);
    if (found) {
        emitted = *found;
        goto type_goes_on_left;
    }

    switch (is_type(type)) {
        case NotAType: assert(false); break;
        case LamType_TAG:
        case BBType_TAG: error("these types do not exist in C");
        case MaskType_TAG: error("should be lowered away");
        case JoinPointType_TAG: error("TODO")
        case NoRet_TAG:
        case Bool_TAG: emitted = "bool"; break;
        case Int_TAG: {
            if (emitter->config.explicitly_sized_types) {
                switch (type->payload.int_type.width) {
                    case IntTy8:  emitted = "uint8_t" ; break;
                    case IntTy16: emitted = "uint16_t"; break;
                    case IntTy32: emitted = "uint32_t"; break;
                    case IntTy64: emitted = "uint64_t"; break;
                }
            } else if (emitter->config.dialect == GLSL) {
                switch (type->payload.int_type.width) {
                    case IntTy8:  warn_print("vanilla GLSL does not support 8-bit integers");
                        emitted = "ubyte";
                        break;
                    case IntTy16: warn_print("vanilla GLSL does not support 16-bit integers");
                        emitted = "ushort";
                        break;
                    case IntTy32: emitted = "uint";   break;
                    case IntTy64: warn_print("vanilla GLSL does not support 64-bit integers");
                        emitted = "ulong";
                        break;
                }
            } else {
                switch (type->payload.int_type.width) {
                    case IntTy8:  emitted = "unsigned char";  break;
                    case IntTy16: emitted = "unsigned short"; break;
                    case IntTy32: emitted = "unsigned int";   break;
                    case IntTy64: emitted = "unsigned long";  break;
                }
            }
            break;
        }
        case Float_TAG: emitted = "float"; break;
        case Type_RecordType_TAG: {
            if (type->payload.record_type.members.count == 0) {
                emitted = "void";
                break;
            }

            emitted = unique_name(emitter->arena, "Record");
            String prefixed = format_string(emitter->arena, "struct %s", emitted);
            emit_nominal_type_body(emitter, prefixed, type);
            // C puts structs in their own namespace so we always need the prefix
            if (emitter->config.dialect == C)
                emitted = prefixed;

            break;
        }
        case Type_QualifiedType_TAG:
            return emit_type(emitter, type->payload.qualified_type.type, center);
        case Type_PtrType_TAG: {
            return emit_type(emitter, type->payload.ptr_type.pointed_type, format_string(emitter->arena, "*%s", center));
        }
        case Type_FnType_TAG: {
            Nodes dom = type->payload.fn_type.param_types;
            Nodes codom = type->payload.fn_type.return_types;

            Growy* paramg = new_growy();
            Printer* paramp = open_growy_as_printer(paramg);
            if (dom.count == 0 && emitter->config.dialect == C)
                print(paramp, "void");
            else for (size_t i = 0; i < dom.count; i++) {
                    print(paramp, emit_type(emitter, dom.nodes[i], NULL));
                    if (i + 1 < dom.count) {
                        print(paramp, ", ");
                    }
                }
            growy_append_bytes(paramg, 1, (char[]) { 0 });
            const char* parameters = printer_growy_unwrap(paramp);
            switch (emitter->config.dialect) {
                case C:
                    center = format_string(emitter->arena, "(%s)(%s)", center, parameters);
                    break;
                case GLSL:
                    // GLSL does not accept functions declared like void (foo)(int);
                    // it also does not support higher-order functions and/or function pointers, so we drop the parentheses
                    center = format_string(emitter->arena, "%s(%s)", center, parameters);
                    break;
            }
            free_tmp_str(parameters);

            return emit_type(emitter, wrap_multiple_yield_types(emitter->arena, codom), center);
        }
        case Type_ArrType_TAG: {
            emitted = unique_name(emitter->arena, "Array");
            String prefixed = format_string(emitter->arena, "struct %s", emitted);
            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);

            print(p, "\n%s {", prefixed);
            indent(p);
            const Node* size = type->payload.arr_type.size;
            String inner_decl_rhs;
            if (size)
                inner_decl_rhs = format_string(emitter->arena, "arr[%s]", emit_value(emitter, size));
            else
                inner_decl_rhs = format_string(emitter->arena, "arr[0]");
            print(p, "\n%s;", emit_type(emitter, type->payload.arr_type.element_type, inner_decl_rhs));
            deindent(p);
            print(p, "\n};\n");
            growy_append_bytes(g, 1, (char[]) { '\0' });

            String subdecl = printer_growy_unwrap(p);
            print(emitter->type_decls, subdecl);
            free_tmp_str(subdecl);

            // ditto from RecordType
            if (emitter->config.dialect == C)
                emitted = prefixed;
            break;
        }
        case Type_PackType_TAG: {
            int width = type->payload.pack_type.width;
            const Type* element_type = type->payload.pack_type.element_type;
            if (emitter->config.dialect == GLSL) {
                assert(is_glsl_scalar_type(element_type));
                assert(width > 1);
                String base;
                switch (element_type->tag) {
                    case Bool_TAG: base = "bvec"; break;
                    case Int_TAG: base = "uvec";  break; // TODO not every int is 32-bit
                    case Float_TAG: base = "vec"; break;
                    default: error("not a valid GLSL vector type");
                }
                emitted = format_string(emitter->arena, "%s%d", base, width);
            } else if (emitter->config.dialect == C) {
                emitted = emit_type(emitter, element_type, NULL);
                emitted = format_string(emitter->arena, "__attribute__ ((vector_size (%d * sizeof(%s) ))) %s", width,
                                        emitted, emitted);
            } else assert(false);
            break;
        }
        case Type_TypeDeclRef_TAG: {
            emit_decl(emitter, type->payload.type_decl_ref.decl);
            emitted = *lookup_existing_type(emitter, type->payload.type_decl_ref.decl);
            goto type_goes_on_left;
        }
    }
    assert(emitted != NULL);
    register_emitted_type(emitter, type, emitted);

    type_goes_on_left:
    assert(emitted != NULL);

    if (strlen(center) > 0)
        emitted = format_string(emitter->arena, "%s %s", emitted, center);

    return emitted;
}
