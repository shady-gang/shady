#include "emit_c.h"

#include "dict.h"
#include "log.h"
#include "util.h"

#include "../shady/ir_private.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#pragma GCC diagnostic error "-Wswitch"

String shd_c_get_record_field_name(Emitter* emitter, const Type* t, size_t i) {
    assert(t->tag == RecordType_TAG);
    RecordType r = t->payload.record_type;
    assert(i < r.members.count);
    if (i >= r.names.count)
        return shd_fmt_string_irarena(t->arena, "_%d", i);
    else
        return shd_c_legalize_identifier(emitter, r.names.strings[i]);
}

static void make_struct_type(Emitter* emitter, String name, Nodes member_types, String member_names[]) {
    Growy* g = shd_new_growy();
    Printer* p = shd_new_printer_from_growy(g);

    shd_print(p, "\n%s {", name);
    shd_printer_indent(p);
    for (size_t i = 0; i < member_types.count; i++) {
        shd_print(p, "\n%s;", shd_c_emit_type(emitter, member_types.nodes[i], member_names[i]));
    }
    shd_printer_deindent(p);
    shd_print(p, "\n};\n");
    shd_growy_append_bytes(g, 1, (char[]) { '\0' });

    shd_print(emitter->type_decls, shd_growy_data(g));
    shd_destroy_growy(g);
    shd_destroy_printer(p);
}

void shd_c_emit_nominal_type_body(Emitter* emitter, String name, const Type* type) {
    assert(type->tag == RecordType_TAG);
    RecordType payload = type->payload.record_type;
    Growy* g = shd_new_growy();
    Printer* p = shd_new_printer_from_growy(g);

    LARRAY(String, member_names, payload.members.count);
    for (size_t i = 0; i < payload.members.count; i++)
        member_names[i] = shd_c_get_record_field_name(emitter, type, i);

    make_struct_type(emitter, name, payload.members, member_names);
}

String shd_c_emit_fn_head(Emitter* emitter, const Node* fn_type, String center, const Node* fn) {
    assert(fn_type->tag == FnType_TAG);
    assert(!fn || fn->type == fn_type);
    Nodes codom = fn_type->payload.fn_type.return_types;

    Growy* paramg = shd_new_growy();
    Printer* paramp = shd_new_printer_from_growy(paramg);
    Nodes dom = fn_type->payload.fn_type.param_types;
    if (dom.count == 0 && emitter->backend_config.dialect == CDialect_C11)
        shd_print(paramp, "void");
    else if (fn) {
        Nodes params = fn->payload.fun.params;
        assert(params.count == dom.count);
        for (size_t i = 0; i < dom.count; i++) {
            String param_name;
            String variable_name = shd_get_node_name_unsafe(fn->payload.fun.params.nodes[i]);
            param_name = shd_fmt_string_irarena(emitter->arena, "%s_%d", shd_c_legalize_identifier(emitter, variable_name), fn->payload.fun.params.nodes[i]->id);
            shd_print(paramp, shd_c_emit_type(emitter, params.nodes[i]->type, param_name));
            if (i + 1 < dom.count) {
                shd_print(paramp, ", ");
            }
        }
    } else {
        for (size_t i = 0; i < dom.count; i++) {
            shd_print(paramp, shd_c_emit_type(emitter, dom.nodes[i], ""));
            if (i + 1 < dom.count) {
                shd_print(paramp, ", ");
            }
        }
    }
    shd_growy_append_bytes(paramg, 1, (char[]) { 0 });
    const char* parameters = shd_printer_growy_unwrap(paramp);
    switch (emitter->backend_config.dialect) {
        default:
            center = shd_format_string_arena(emitter->arena->arena, "(%s)(%s)", center, parameters);
            break;
        case CDialect_GLSL:
            // GLSL does not accept functions declared like void (foo)(int);
            // it also does not support higher-order functions and/or function pointers, so we drop the parentheses
            center = shd_format_string_arena(emitter->arena->arena, "%s(%s)", center, parameters);
            break;
    }
    free_tmp_str(parameters);

    String c_decl = shd_c_emit_type(emitter, shd_maybe_multiple_return(emitter->arena, codom), center);

    return c_decl;
}

String shd_c_emit_dim(SpvDim dim_id) {
    String dim;
    switch (dim_id) {
        case SpvDim1D:
            dim = "1D";
            break;
        case SpvDim2D:
            dim = "2D";
            break;
        case SpvDim3D:
            dim = "3D";
            break;
        case SpvDimCube:
            dim = "Cube";
            break;
        case SpvDimRect:
            dim = "Rect";
            break;
        case SpvDimBuffer:
            dim = "Buffer";
            break;
        case SpvDimSubpassData:
        case SpvDimTileImageDataEXT:
        case SpvDimMax:
        shd_error("Invalid Dim");
    }
    return dim;
}

static String cuda_int_type(IrArena* a, Int type, int vector_width) {
    const char* ispc_int_types[4][2] = {
        { "uchar" , "char"  },
        { "ushort", "short" },
        { "uint", "int" },
        { "ulong", "long" },
    };
    return shd_fmt_string_irarena(a, "%s%d", ispc_int_types[type.width][type.is_signed], vector_width);
}

String shd_c_emit_type(Emitter* emitter, const Type* type, const char* center) {
    if (center == NULL)
        center = "";

    String emitted = NULL;
    CType* found = shd_c_lookup_existing_type(emitter, type);
    if (found) {
        emitted = *found;
        goto type_goes_on_left;
    }

    switch (is_type(type)) {
        case NotAType: assert(false); break;
        case LamType_TAG:
        case BBType_TAG: shd_error("these types do not exist in C");
        case Type_SampledImageType_TAG: {
            if (emitter->backend_config.dialect != CDialect_GLSL) {
                shd_error("TODO: implement textures on non-glsl backends");
            }

            const Type* sampled = type->payload.sampled_image_type.image_type;
            assert(sampled->tag == ImageType_TAG);

            String prefix = "";
            if (sampled->payload.image_type.sampled_type->tag == Int_TAG)
                prefix = sampled->payload.image_type.sampled_type->payload.int_type.is_signed ? "i" : "u";

            String dim = shd_c_emit_dim(sampled->payload.image_type.dim);

            String array = sampled->payload.image_type.arrayed ? "Array" : "";
            String shadow = sampled->payload.image_type.depth ? "Shadow" : "";

            emitted = shd_fmt_string_irarena(emitter->arena, "%ssampler%s%s%s", prefix, dim, array, shadow);
            break;
        }
        case Type_SamplerType_TAG:
        case Type_ImageType_TAG:
        case JoinPointType_TAG: shd_error("TODO")
        case NoRet_TAG:
        case Bool_TAG: emitted = "bool"; break;
        case Int_TAG: {
            bool sign = type->payload.int_type.is_signed;
            switch (emitter->backend_config.dialect) {
                case CDialect_ISPC: {
                    const char* ispc_int_types[4][2] = {
                        { "uint8" , "int8"  },
                        { "uint16", "int16" },
                        { "uint32", "int32" },
                        { "uint64", "int64" },
                    };
                    emitted = ispc_int_types[type->payload.int_type.width][sign];
                    break;
                }
                case CDialect_CUDA:
                //    emitted = cuda_int_type(emitter->arena, type->payload.int_type, 1);
                //    break;
                case CDialect_C11: {
                    const char* c_classic_int_types[4][2] = {
                            { "unsigned char" , "char"  },
                            { "unsigned short", "short" },
                            { "unsigned int"  , "int" },
                            { "unsigned long long" , "long long" },
                    };
                    const char* c_explicit_int_sizes[4][2] = {
                            { "uint8_t" , "int8_t"  },
                            { "uint16_t", "int16_t" },
                            { "uint32_t", "int32_t" },
                            { "uint64_t", "int64_t" },
                    };
                    emitted = (emitter->backend_config.explicitly_sized_types ? c_explicit_int_sizes : c_classic_int_types)[type->payload.int_type.width][sign];
                    break;
                }
                case CDialect_GLSL:
                    if (emitter->backend_config.glsl_version <= 120) {
                        emitted = "int";
                        break;
                    }
                    switch (type->payload.int_type.width) {
                        case ShdIntSize8:  shd_warn_print("vanilla GLSL does not support 8-bit integers\n");
                            emitted = sign ? "byte" : "ubyte";
                            break;
                        case ShdIntSize16: shd_warn_print("vanilla GLSL does not support 16-bit integers\n");
                            emitted = sign ? "short" : "ushort";
                            break;
                        case ShdIntSize32: emitted = sign ? "int" : "uint"; break;
                        case ShdIntSize64:
                            emitter->need_64b_ext = true;
                            shd_warn_print("vanilla GLSL does not support 64-bit integers\n");
                            emitted = sign ? "int64_t" : "uint64_t";
                            break;
                    }
                    break;
            }
            break;
        }
        case Float_TAG:
            switch (type->payload.float_type.width) {
                case ShdFloatFormat16:
                    assert(false);
                    break;
                case ShdFloatFormat32:
                    emitted = "float";
                    break;
                case ShdFloatFormat64:
                    emitted = "double";
                    break;
            }
            break;
        case Type_RecordType_TAG: {
            emitted = shd_make_unique_name(emitter->arena, "Record");
            String prefixed = shd_format_string_arena(emitter->arena->arena, "struct %s", emitted);
            shd_c_emit_nominal_type_body(emitter, prefixed, type);
            // C puts structs in their own namespace so we always need the prefix
            if (emitter->backend_config.dialect == CDialect_C11)
                emitted = prefixed;

            break;
        }
        case Type_TupleType_TAG: {
            TupleType payload = type->payload.tuple_type;
            if (payload.members.count == 0) {
                emitted = "void";
                break;
            }

            emitted = shd_make_unique_name(emitter->arena, "Tuple");
            String prefixed = shd_format_string_arena(emitter->arena->arena, "struct %s", emitted);
            LARRAY(String, member_names, payload.members.count);
            for (unsigned i = 0; i < payload.members.count; i++)
                member_names[i] = shd_format_string_arena(emitter->arena->arena, "_%d", i);
            make_struct_type(emitter, prefixed, payload.members, member_names);
            // C puts structs in their own namespace so we always need the prefix
            if (emitter->backend_config.dialect == CDialect_C11)
                emitted = prefixed;

            break;
        }
        case Type_QualifiedType_TAG:
            if (type->payload.qualified_type.type == unit_type(emitter->arena)) {
                emitted = "void";
                break;
            }
            switch (emitter->backend_config.dialect) {
                default:
                    return shd_c_emit_type(emitter, type->payload.qualified_type.type, center);
                case CDialect_ISPC:
                    if (type->payload.qualified_type.scope <= ShdScopeSubgroup)
                        return shd_c_emit_type(emitter, type->payload.qualified_type.type, shd_format_string_arena(emitter->arena->arena, "uniform %s", center));
                    else
                        return shd_c_emit_type(emitter, type->payload.qualified_type.type, shd_format_string_arena(emitter->arena->arena, "varying %s", center));
            }
        case Type_PtrType_TAG: {
            CType t = shd_c_emit_type(emitter, type->payload.ptr_type.pointed_type, shd_format_string_arena(emitter->arena->arena, "* %s", center));
            if (emitter->backend_config.dialect == CDialect_ISPC) {
                ShdScope scope = shd_get_addr_space_scope(type->payload.ptr_type.address_space);
                t = shd_format_string_arena(emitter->arena->arena, "%s %s", scope > ShdScopeSubgroup ? "varying" : "uniform", t);
            }
            return t;
        }
        case Type_FnType_TAG: {
            return shd_c_emit_fn_head(emitter, type, center, NULL);
        }
        case Type_ArrType_TAG: {
            const Node* size = type->payload.arr_type.size;
            emitted = shd_make_unique_name(emitter->arena, size ? "Array" : "UnsizedArray");
            String prefixed = shd_format_string_arena(emitter->arena->arena, "struct %s", emitted);
            Growy* g = shd_new_growy();
            Printer* p = shd_new_printer_from_growy(g);

            if (!size && emitter->backend_config.decay_unsized_arrays)
                return shd_c_emit_type(emitter, type->payload.arr_type.element_type, center);

            shd_print(p, "\n%s {", prefixed);
            shd_printer_indent(p);
            String inner_decl_rhs;
            if (size)
                inner_decl_rhs = shd_format_string_arena(emitter->arena->arena, "arr[%zu]", shd_get_int_literal_value(*shd_resolve_to_int_literal(size), false));
            else
                inner_decl_rhs = shd_format_string_arena(emitter->arena->arena, "arr[1]");
            shd_print(p, "\n%s;", shd_c_emit_type(emitter, type->payload.arr_type.element_type, inner_decl_rhs));
            shd_printer_deindent(p);
            shd_print(p, "\n};\n");
            shd_growy_append_bytes(g, 1, (char[]) { '\0' });

            String subdecl = shd_printer_growy_unwrap(p);
            shd_print(emitter->type_decls, subdecl);
            free_tmp_str(subdecl);

            // ditto from RecordType
            switch (emitter->backend_config.dialect) {
                default:
                    emitted = prefixed;
                    break;
                case CDialect_GLSL:
                    break;
            }
            break;
        }
        case Type_VectorType_TAG: {
            int width = type->payload.vector_type.width;
            const Type* element_type = type->payload.vector_type.element_type;
            switch (emitter->backend_config.dialect) {
                case CDialect_CUDA:
                    assert(element_type->tag == Int_TAG);
                    emitted = cuda_int_type(emitter->arena, element_type->payload.int_type, width);
                    break;
                case CDialect_GLSL: {
                    assert(is_glsl_scalar_type(element_type));
                    assert(width > 1);
                    String base;
                    switch (element_type->tag) {
                        case Bool_TAG: base = "bvec"; break;
                        case Int_TAG: base = element_type->payload.int_type.is_signed ? "ivec": "uvec"; break; // TODO not every int is 32-bit
                        case Float_TAG: base = "vec"; break;
                        default: shd_error("not a valid GLSL vector type");
                    }
                    emitted = shd_format_string_arena(emitter->arena->arena, "%s%d", base, width);
                    break;
                }
                case CDialect_ISPC: shd_error("Please lower to something else")
                case CDialect_C11: {
                    emitted = shd_c_emit_type(emitter, element_type, NULL);
                    emitted = shd_format_string_arena(emitter->arena->arena, "__attribute__ ((vector_size (%d * sizeof(%s) ))) %s", width, emitted, emitted);
                    break;
                }
            }
            break;
        }
        case Type_MatrixType_TAG: {
            int columns = type->payload.matrix_type.columns;
            const Type* element_type = type->payload.matrix_type.element_type;
            int rows = shd_deconstruct_vector_type(&element_type);
            switch (emitter->backend_config.dialect) {
                case CDialect_CUDA:
                    assert(element_type->tag == Int_TAG);
                    emitted = cuda_int_type(emitter->arena, element_type->payload.int_type, columns * rows);
                    break;
                case CDialect_GLSL: {
                    assert(is_glsl_scalar_type(element_type));
                    assert(columns > 1);
                    String base;
                    switch (element_type->tag) {
                        // this is pure fiction, glsl doesn't have non-float matrices ... yet ?
                        case Bool_TAG: base = "bmat"; break;
                        case Int_TAG: base = element_type->payload.int_type.is_signed ? "imat": "umat"; break; // TODO not every int is 32-bit
                        case Float_TAG: base = "mat"; break;
                        default: shd_error("not a valid GLSL vector type");
                    }
                    if (columns == rows)
                        emitted = shd_format_string_arena(emitter->arena->arena, "%s%d", base, columns);
                    else
                        emitted = shd_format_string_arena(emitter->arena->arena, "%s%dx%d", base, columns, rows);
                    break;
                }
                case CDialect_ISPC: shd_error("Please lower to something else")
                case CDialect_C11: {
                    emitted = shd_c_emit_type(emitter, element_type, NULL);
                    emitted = shd_format_string_arena(emitter->arena->arena, "__attribute__ ((vector_size (%d * %d * sizeof(%s) ))) %s", columns, rows, emitted, emitted);
                    break;
                }
            }
            break;
        }
        case NominalType_TAG: {
            shd_c_emit_decl(emitter, type);
            emitted = *shd_c_lookup_existing_type(emitter, type);
            goto type_goes_on_left;
        }
        case Type_ExtType_TAG: {
            shd_error("TODO") // matrices etc
        }
    }
    assert(emitted != NULL);
    shd_c_register_emitted_type(emitter, type, emitted);

    type_goes_on_left:
    assert(emitted != NULL);

    if (strlen(center) > 0)
        emitted = shd_format_string_arena(emitter->arena->arena, "%s %s", emitted, center);

    return emitted;
}
