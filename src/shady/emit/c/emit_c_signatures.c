#include "emit_c.h"

#include "dict.h"
#include "log.h"
#include "util.h"

#include "../../type.h"
#include "../../ir_private.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

String c_get_record_field_name(const Type* t, size_t i) {
    assert(t->tag == RecordType_TAG);
    RecordType r = t->payload.record_type;
    assert(i < r.members.count);
    if (i >= r.names.count)
        return format_string_interned(t->arena, "_%d", i);
    else
        return r.names.strings[i];
}

void c_emit_nominal_type_body(Emitter* emitter, String name, const Type* type) {
    assert(type->tag == RecordType_TAG);
    Growy* g = new_growy();
    Printer* p = open_growy_as_printer(g);

    print(p, "\n%s {", name);
    indent(p);
    for (size_t i = 0; i < type->payload.record_type.members.count; i++) {
        String member_identifier = c_get_record_field_name(type, i);
        print(p, "\n%s;", c_emit_type(emitter, type->payload.record_type.members.nodes[i], member_identifier));
    }
    deindent(p);
    print(p, "\n};\n");
    growy_append_bytes(g, 1, (char[]) { '\0' });

    print(emitter->type_decls, growy_data(g));
    destroy_growy(g);
    destroy_printer(p);
}

String c_emit_fn_head(Emitter* emitter, const Node* fn_type, String center, const Node* fn) {
    assert(fn_type->tag == FnType_TAG);
    assert(!fn || fn->type == fn_type);
    Nodes codom = fn_type->payload.fn_type.return_types;

    Growy* paramg = new_growy();
    Printer* paramp = open_growy_as_printer(paramg);
    Nodes dom = fn_type->payload.fn_type.param_types;
    if (dom.count == 0 && emitter->config.dialect == C)
        print(paramp, "void");
    else if (fn) {
        Nodes params = fn->payload.fun.params;
        assert(params.count == dom.count);
        for (size_t i = 0; i < dom.count; i++) {
            String param_name;
            String variable_name = get_value_name(fn->payload.fun.params.nodes[i]);
            param_name = unique_name(emitter->arena, legalize_c_identifier(emitter, variable_name));
            print(paramp, c_emit_type(emitter, params.nodes[i]->type, param_name));
            if (i + 1 < dom.count) {
                print(paramp, ", ");
            }
        }
    } else {
        for (size_t i = 0; i < dom.count; i++) {
            print(paramp, c_emit_type(emitter, dom.nodes[i], ""));
            if (i + 1 < dom.count) {
                print(paramp, ", ");
            }
        }
    }
    growy_append_bytes(paramg, 1, (char[]) { 0 });
    const char* parameters = printer_growy_unwrap(paramp);
    switch (emitter->config.dialect) {
        case ISPC:
        case C:
            center = format_string_arena(emitter->arena->arena, "(%s)(%s)", center, parameters);
            break;
        case GLSL:
            // GLSL does not accept functions declared like void (foo)(int);
            // it also does not support higher-order functions and/or function pointers, so we drop the parentheses
            center = format_string_arena(emitter->arena->arena, "%s(%s)", center, parameters);
            break;
    }
    free_tmp_str(parameters);

    String c_decl = c_emit_type(emitter, wrap_multiple_yield_types(emitter->arena, codom), center);

    const Node* entry_point = fn ? lookup_annotation(fn, "EntryPoint") : NULL;
    if (entry_point) switch (emitter->config.dialect) {
            case C:
                break;
            case GLSL:
                break;
            case ISPC:
                c_decl = format_string_arena(emitter->arena->arena, "export %s", c_decl);
                break;
        }

    return c_decl;
}

String c_emit_type(Emitter* emitter, const Type* type, const char* center) {
    if (center == NULL)
        center = "";

    String emitted = NULL;
    CType* found = lookup_existing_ctype(emitter, type);
    if (found) {
        emitted = *found;
        goto type_goes_on_left;
    }

    switch (is_type(type)) {
        case NotAType: assert(false); break;
        case LamType_TAG:
        case BBType_TAG: error("these types do not exist in C");
        case MaskType_TAG: error("should be lowered away");
        case Type_SampledImageType_TAG:
        case Type_SamplerType_TAG:
        case Type_ImageType_TAG:
        case JoinPointType_TAG: error("TODO")
        case NoRet_TAG:
        case Bool_TAG: emitted = "bool"; break;
        case Int_TAG: {
            switch (emitter->config.dialect) {
                case ISPC: {
                    const char* ispc_int_types[4][2] = {
                        { "uint8" , "int8"  },
                        { "uint16", "int16" },
                        { "uint32", "int32" },
                        { "uint64", "int64" },
                    };
                    emitted = ispc_int_types[type->payload.int_type.width][type->payload.int_type.is_signed];
                    break;
                }
                case C: {
                    const char* c_classic_int_types[4][2] = {
                            { "unsigned char" , "char"  },
                            { "unsigned short", "short" },
                            { "unsigned int"  , "int" },
                            { "unsigned long" , "long" },
                    };
                    const char* c_explicit_int_sizes[4][2] = {
                            { "uint8_t" , "int8_t"  },
                            { "uint16_t", "int16_t" },
                            { "uint32_t", "int32_t" },
                            { "uint64_t", "int64_t" },
                    };
                    emitted = (emitter->config.explicitly_sized_types ? c_explicit_int_sizes : c_classic_int_types)[type->payload.int_type.width][type->payload.int_type.is_signed];
                    break;
                }
                case GLSL:
                    switch (type->payload.int_type.width) {
                        case IntTy8:  warn_print("vanilla GLSL does not support 8-bit integers\n");
                            emitted = "ubyte";
                            break;
                        case IntTy16: warn_print("vanilla GLSL does not support 16-bit integers\n");
                            emitted = "ushort";
                            break;
                        case IntTy32: emitted = "uint";   break;
                        case IntTy64: warn_print("vanilla GLSL does not support 64-bit integers\n");
                            emitted = "uint64_t";
                            break;
                    }
                    break;
            }
            break;
        }
        case Float_TAG:
            switch (type->payload.float_type.width) {
                case FloatTy16:
                    assert(false);
                    break;
                case FloatTy32:
                    emitted = "float";
                    break;
                case FloatTy64:
                    emitted = "double";
                    break;
            }
            break;
        case Type_RecordType_TAG: {
            if (type->payload.record_type.members.count == 0) {
                emitted = "void";
                break;
            }

            emitted = unique_name(emitter->arena, "Record");
            String prefixed = format_string_arena(emitter->arena->arena, "struct %s", emitted);
            c_emit_nominal_type_body(emitter, prefixed, type);
            // C puts structs in their own namespace so we always need the prefix
            if (emitter->config.dialect == C)
                emitted = prefixed;

            break;
        }
        case Type_QualifiedType_TAG:
            switch (emitter->config.dialect) {
                case C:
                case GLSL:
                    return c_emit_type(emitter, type->payload.qualified_type.type, center);
                case ISPC:
                    if (type->payload.qualified_type.is_uniform)
                        return c_emit_type(emitter, type->payload.qualified_type.type, format_string_arena(emitter->arena->arena, "uniform %s", center));
                    else
                        return c_emit_type(emitter, type->payload.qualified_type.type, format_string_arena(emitter->arena->arena, "varying %s", center));
            }
        case Type_PtrType_TAG: {
            CType t = c_emit_type(emitter, type->payload.ptr_type.pointed_type, format_string_arena(emitter->arena->arena, "* %s", center));
            // we always emit pointers to _uniform_ data, no exceptions
            if (emitter->config.dialect == ISPC)
                t = format_string_arena(emitter->arena->arena, "uniform %s", t);
            return t;
        }
        case Type_FnType_TAG: {
            return c_emit_fn_head(emitter, type, center, NULL);
        }
        case Type_ArrType_TAG: {
            emitted = unique_name(emitter->arena, "Array");
            String prefixed = format_string_arena(emitter->arena->arena, "struct %s", emitted);
            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);

            print(p, "\n%s {", prefixed);
            indent(p);
            const Node* size = type->payload.arr_type.size;
            String inner_decl_rhs;
            if (size)
                inner_decl_rhs = format_string_arena(emitter->arena->arena, "arr[%zu]", get_int_literal_value(*resolve_to_int_literal(size), false));
            else
                inner_decl_rhs = format_string_arena(emitter->arena->arena, "arr[0]");
            print(p, "\n%s;", c_emit_type(emitter, type->payload.arr_type.element_type, inner_decl_rhs));
            deindent(p);
            print(p, "\n};\n");
            growy_append_bytes(g, 1, (char[]) { '\0' });

            String subdecl = printer_growy_unwrap(p);
            print(emitter->type_decls, subdecl);
            free_tmp_str(subdecl);

            // ditto from RecordType
            switch (emitter->config.dialect) {
                case C:
                case ISPC:
                    emitted = prefixed;
                    break;
                case GLSL:
                    break;
            }
            break;
        }
        case Type_PackType_TAG: {
            int width = type->payload.pack_type.width;
            const Type* element_type = type->payload.pack_type.element_type;
            switch (emitter->config.dialect) {
                case GLSL: {
                    assert(is_glsl_scalar_type(element_type));
                    assert(width > 1);
                    String base;
                    switch (element_type->tag) {
                        case Bool_TAG: base = "bvec"; break;
                        case Int_TAG: base = "uvec";  break; // TODO not every int is 32-bit
                        case Float_TAG: base = "vec"; break;
                        default: error("not a valid GLSL vector type");
                    }
                    emitted = format_string_arena(emitter->arena->arena, "%s%d", base, width);
                    break;
                }
                case ISPC: error("Please lower to something else")
                case C: {
                    emitted = c_emit_type(emitter, element_type, NULL);
                    emitted = format_string_arena(emitter->arena->arena, "__attribute__ ((vector_size (%d * sizeof(%s) ))) %s", width, emitted, emitted);
                    break;
                }
            }
            break;
        }
        case Type_TypeDeclRef_TAG: {
            c_emit_decl(emitter, type->payload.type_decl_ref.decl);
            emitted = *lookup_existing_ctype(emitter, type->payload.type_decl_ref.decl);
            goto type_goes_on_left;
        }
    }
    assert(emitted != NULL);
    register_emitted_ctype(emitter, type, emitted);

    type_goes_on_left:
    assert(emitted != NULL);

    if (strlen(center) > 0)
        emitted = format_string_arena(emitter->arena->arena, "%s %s", emitted, center);

    return emitted;
}
