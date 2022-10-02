#include "emit_c.h"

#include "dict.h"
#include "log.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef const char* String;

#pragma GCC diagnostic error "-Wswitch"

static String append_unique_number(Emitter* emitter, String name) {
    unsigned id = emitter->next_id++;
    String formatted = format_string(emitter->arena, "%s_%d", name, id);
    return formatted;
}

static const Type* codom_to_single_type(IrArena* arena, Nodes codom) {
    switch (codom.count) {
        case 0: return unit_type(arena);
        case 1: return codom.nodes[0];
        default: {
            const Type* codom_ret_type = record_type(arena, (RecordType) { .members = codom, .special = MultipleReturn });
            return codom_ret_type;
        }
    }
}

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
        case MaskType_TAG: error("should be lowered away");
        case Type_NoRet_TAG:
        case Type_Unit_TAG: emitted = "void"; break;
        case Bool_TAG: emitted = "bool"; break;
        case Int_TAG: {
            if (emitter->config.explicitly_sized_types) {
                switch (type->payload.int_type.width) {
                    case IntTy8:  emitted = "int8_t" ; break;
                    case IntTy16: emitted = "int16_t"; break;
                    case IntTy32: emitted = "int32_t"; break;
                    case IntTy64: emitted = "int64_t"; break;
                }
            } else {
                switch (type->payload.int_type.width) {
                    case IntTy8:  emitted = "char";  break;
                    case IntTy16: emitted = "short"; break;
                    case IntTy32: emitted = "int";   break;
                    case IntTy64: emitted = "long";  break;
                }
            }
            break;
        }
        case Float_TAG: emitted = "float"; break;
        case Type_RecordType_TAG: {
            emitted = append_unique_number(emitter, "struct Record");
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
            growy_append_bytes(g, 1, (char[]) { '\0' });
            deindent(p);
            print(p, "\n};\n");

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
            assert(!type->payload.fn_type.is_basic_block);
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
            const char* parameters = printer_growy_unwrap(paramp);
            center = format_string(emitter->arena, "(%s)(%s)", center, parameters);
            free(parameters);

            return emit_type(emitter, codom_to_single_type(emitter->arena, codom), center);
        }
        case Type_ArrType_TAG: {
            const Node* size = type->payload.arr_type.size;
            if (size)
                center = format_string(emitter->arena, "%s[%s]", center, emit_value(emitter, size));
            else
                center = format_string(emitter->arena, "%s[]", center);
            return emit_type(emitter, type->payload.arr_type.element_type, center);
        }
        case Type_PackType_TAG: error("TODO");
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

String emit_value(Emitter* emitter, const Node* value) {
    String* found = find_value_dict(const Node*, String, emitter->emitted, value);
    if (found)
        return *found;

    String emitted = NULL;

    switch (is_value(value)) {
        case NotAValue: assert(false);
        case Value_Variable_TAG: error("variables need to be emitted beforehand");
        case Value_Unbound_TAG: break;
        case Value_UntypedNumber_TAG: break;
        case Value_IntLiteral_TAG: emitted = format_string(emitter->arena, "%d", value->payload.int_literal.value_u64); break;
        case Value_True_TAG: return "true";
        case Value_False_TAG: return "false";
        case Value_StringLiteral_TAG: break;
        case Value_ArrayLiteral_TAG: break;
        case Value_Tuple_TAG: break;
        case Value_FnAddr_TAG: break;
        case Value_RefDecl_TAG: break;
    }

    assert(emitted);
    return emitted;
}

static void emit_instruction(Emitter* emitter, const Node* instruction);

static String emit_decl(Emitter* emitter, const Node* decl) {
    String* found = find_value_dict(const Node*, String, emitter->emitted, decl);
    if (found)
        return *found;

    assert(is_declaration(decl->tag));
    switch (decl->tag) {
        case GlobalVariable_TAG:

        case Function_TAG:
        case Constant_TAG:
        default: error("not a decl");
    }
}

KeyHash hash_node(Node**);
bool compare_node(Node**, Node**);

void emit_c(CompilerConfig* config, IrArena* arena, const Node* root_node, size_t* output_size, char** output) {
    Growy* type_decls_g = new_growy();
    Growy* fn_decls_g = new_growy();
    Growy* fn_defs_g = new_growy();

    Emitter emitter = {
        .config = {
            .config = config,
            .dialect = C,
        },
        .type_decls = open_growy_as_printer(type_decls_g),
        .fn_decls = open_growy_as_printer(fn_decls_g),
        .fn_defs = open_growy_as_printer(fn_defs_g),
        .emitted = new_dict(Node*, String, (HashFn) hash_node, (CmpFn) compare_node)
    };

    assert(root_node->tag == Root_TAG);
    Nodes decls = root_node->payload.root.declarations;
    for (size_t i = 0; i < decls.count; i++)
        emit_decl(&emitter, decls.nodes[i]);

    Growy* final = new_growy();
    Printer* finalp = open_growy_as_printer(final);

    print(finalp, "// file generated by shady \n");

    destroy_printer(emitter.type_decls);
    destroy_printer(emitter.fn_decls);
    destroy_printer(emitter.fn_defs);

    growy_append_bytes(final, growy_size(type_decls_g), growy_data(type_decls_g));
    growy_append_bytes(final, growy_size(fn_decls_g), growy_data(fn_decls_g));
    growy_append_bytes(final, growy_size(fn_defs_g), growy_data(fn_defs_g));

    growy_destroy(type_decls_g);
    growy_destroy(fn_decls_g);
    growy_destroy(fn_defs_g);

    destroy_dict(emitter.emitted);

    *output_size = growy_size(final);
    *output = growy_deconstruct(final);
    destroy_printer(finalp);

    fwrite(*output, *output_size, 1, stderr);
}
