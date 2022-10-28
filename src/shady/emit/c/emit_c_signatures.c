#include "emit_c.h"

#include "dict.h"
#include "log.h"

#include "../../type.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

String emit_type(Emitter* emitter, const Type* type, const char* center) {
    if (center == NULL)
        center = "";

    String emitted = NULL;
    String* found = find_value_dict(const Node*, String, emitter->emitted, type);
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
        case Unit_TAG: emitted = "void"; break;
        case Bool_TAG: emitted = "bool"; break;
        case Int_TAG: {
            if (emitter->config.explicitly_sized_types) {
                switch (type->payload.int_type.width) {
                    case IntTy8:  emitted = "unt8_t" ; break;
                    case IntTy16: emitted = "unt16_t"; break;
                    case IntTy32: emitted = "unt32_t"; break;
                    case IntTy64: emitted = "unt64_t"; break;
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
            emitted = unique_name(emitter->arena, "struct Record");
            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);

            print(p, "\n%s {", emitted);
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
            growy_destroy(g);
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
            if (dom.count == 0)
                print(paramp, "void");
            else for (size_t i = 0; i < dom.count; i++) {
                    print(paramp, emit_type(emitter, dom.nodes[i], NULL));
                    if (i + 1 < dom.count) {
                        print(paramp, ", ");
                    }
                }
            growy_append_bytes(paramg, 1, (char[]) { 0 });
            const char* parameters = printer_growy_unwrap(paramp);
            center = format_string(emitter->arena, "(%s)(%s)", center, parameters);
            free_tmp_str(parameters);

            return emit_type(emitter, wrap_multiple_yield_types(emitter->arena, codom), center);
        }
        case Type_ArrType_TAG: {
            emitted = unique_name(emitter->arena, "struct Array");
            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);

            print(p, "\n%s {", emitted);
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
            break;
        }
        case Type_PackType_TAG: {
            emitted = emit_type(emitter, type->payload.pack_type.element_type, NULL);
            emitted = format_string(emitter->arena, "__attribute__ ((vector_size (%d * sizeof(%s) ))) %s", type->payload.pack_type.width, emitted, emitted);
            break;
        }
        case Type_NominalType_TAG: {
            emitted = type->payload.nom_type.name;
            insert_dict(const Node*, String, emitter->emitted, type, emitted);
            print(emitter->type_decls, "\ntypedef %s;", emit_type(emitter, type->payload.nom_type.body, emitted));
            goto type_goes_on_left;
        }
    }
    assert(emitted != NULL);
    insert_dict(const Node*, String, emitter->emitted, type, emitted);

    type_goes_on_left:
    assert(emitted != NULL);

    if (strlen(center) > 0)
        emitted = format_string(emitter->arena, "%s %s", emitted, center);

    return emitted;
}
