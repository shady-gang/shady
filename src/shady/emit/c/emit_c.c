#include "emit_c.h"

#include "portability.h"
#include "dict.h"
#include "log.h"

#include "../../type.h"
#include "../../ir_private.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef const char* String;

#pragma GCC diagnostic error "-Wswitch"

static String emit_decl(Emitter* emitter, const Node* decl);
static String emit_block_helper(Emitter* emitter, const Node* block, const Nodes* bbs);

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
            growy_append_bytes(paramg, 1, (char[]) { 0 });
            const char* parameters = printer_growy_unwrap(paramp);
            center = format_string(emitter->arena, "(%s)(%s)", center, parameters);
            free(parameters);

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
            free(subdecl);
            break;

            /*const Node* size = type->payload.arr_type.size;
            if (size)
                center = format_string(emitter->arena, "%s[%s]", center, emit_value(emitter, size));
            else
                center = format_string(emitter->arena, "%s[]", center);
            return emit_type(emitter, type->payload.arr_type.element_type, center);*/
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

String emit_fn_head(Emitter* emitter, const Node* fn) {
    assert(fn->tag == Function_TAG);
    Nodes dom = fn->payload.fn.params;
    Nodes codom = fn->payload.fn.return_types;

    Growy* paramg = new_growy();
    Printer* paramp = open_growy_as_printer(paramg);
    if (dom.count == 0)
        print(paramp, "void");
    else for (size_t i = 0; i < dom.count; i++) {
        print(paramp, emit_type(emitter, dom.nodes[i]->type, format_string(emitter->arena, "%s_%d", dom.nodes[i]->payload.var.name, dom.nodes[i]->payload.var.id)));
        if (i + 1 < dom.count) {
            print(paramp, ", ");
        }
    }
    growy_append_bytes(paramg, 1, (char[]) { 0 });
    const char* parameters = printer_growy_unwrap(paramp);
    String center = format_string(emitter->arena, "%s(%s)", fn->payload.fn.name, parameters);
    free(parameters);

    return emit_type(emitter, wrap_multiple_yield_types(emitter->arena, codom), center);
}

#include <ctype.h>

static enum { ObjectsList, StringLit, CharsLit } array_insides_helper(Emitter* e, Printer* p, Growy* g, const Node* array) {
    const Type* t = array->payload.arr_lit.element_type;
    Nodes c = array->payload.arr_lit.contents;
    if (t->tag == Int_TAG && t->payload.int_type.width == 8) {
        uint8_t* tmp = malloc(sizeof(uint8_t) * c.count);
        bool ends_zero = false, has_zero = false;
        for (size_t i = 0; i < c.count; i++) {
            tmp[i] = extract_int_literal_value(c.nodes[i], false);
            if (tmp[i] == 0) {
                if (i == c.count - 1)
                    ends_zero = true;
                else
                    has_zero = true;
            }
        }
        bool is_stringy = ends_zero /*&& !has_zero*/;
        for (size_t i = 0; i < c.count; i++) {
            // ignore the last char in a string
            if (is_stringy && i == c.count - 1)
                break;
            if (isprint(tmp[i]))
                print(p, "%c", tmp[i]);
            else
                print(p, "\\x%02x", tmp[i]);
        }
        free(tmp);
        return is_stringy ? StringLit : CharsLit;
    } else {
        for (size_t i = 0; i < c.count; i++) {
            print(p, emit_value(e, c.nodes[i]));
            if (i + 1 < c.count)
                print(p, ", ");
        }
        growy_append_bytes(g, 1, "\0");
        return ObjectsList;
    }
}

String emit_value(Emitter* emitter, const Node* value) {
    String* found = find_value_dict(const Node*, String, emitter->emitted, value);
    if (found)
        return *found;

    String emitted = NULL;

    switch (is_value(value)) {
        case NotAValue: assert(false);
        case Value_Unbound_TAG:
        case Value_UntypedNumber_TAG: error("lower me");
        case Value_Variable_TAG: error("variables need to be emitted beforehand");
        case Value_IntLiteral_TAG: emitted = format_string(emitter->arena, "%d", value->payload.int_literal.value_u64); break;
        case Value_True_TAG: return "true";
        case Value_False_TAG: return "false";
        case Value_Tuple_TAG: break;
        case Value_StringLiteral_TAG: break;
        case Value_ArrayLiteral_TAG: {
            Growy* g = new_growy();
            Printer* p = open_growy_as_printer(g);
            switch (array_insides_helper(emitter, p, g, value)) {
                case ObjectsList:
                    emitted = format_string(emitter->arena, "((%s) { %s })", emit_type(emitter, value->payload.arr_lit.element_type, NULL), growy_data(g));
                    break;
                case StringLit:
                    emitted = format_string(emitter->arena, "((%s) { \"s\" })", emit_type(emitter, value->payload.arr_lit.element_type, NULL), growy_data(g));
                    break;
                case CharsLit:
                    emitted = format_string(emitter->arena, "((%s) { '%s' })", emit_type(emitter, value->payload.arr_lit.element_type, NULL), growy_data(g));
                    break;
            }
            growy_destroy(g);
            destroy_printer(p);
            break;
        }
        case Value_FnAddr_TAG: {
            emitted = get_decl_name(value->payload.fn_addr.fn);
            emitted = format_string(emitter->arena, "&%s", emitted);
            break;
        }
        case Value_RefDecl_TAG: {
            emitted = emit_decl(emitter, value->payload.ref_decl.decl);
            break;
        }
    }

    assert(emitted);
    return emitted;
}

static void declare_variables_helper(Emitter* emitter, Printer* p, const Nodes* vars) {
    for (size_t i = 0; i < vars->count; i++) {
        const Variable* var = &vars->nodes[i]->payload.var;
        String named = format_string(emitter->arena, "%s_%d", var->name, var->id);
        print(p, "\n%s;", c_emit_type(emitter, var->type, named));
        insert_dict(const Node*, String, emitter->emitted, vars->nodes[i], named);
    }
}

static void emit_primop(Emitter* emitter, Printer* p, const PrimOp* prim_op, const Nodes* outputs) {
    enum {
        Infix, Prefix
    } m = Infix;
    String s = NULL, rhs = NULL;
    switch (prim_op->op) {
        case add_op: s = "+";  break;
        case sub_op: s = "-";  break;
        case mul_op: s = "*";  break;
        case div_op: s = "/";  break;
        case mod_op: s = "%";  break;
        case neg_op: s = "-"; m = Prefix; break;
        case gt_op:  s = ">";  break;
        case gte_op: s = ">="; break;
        case lt_op:  s = "<";  break;
        case lte_op: s = "<="; break;
        case eq_op:  s = "=="; break;
        case neq_op: s = "!="; break;
        case and_op: s = "&";  break;
        case or_op:  s = "|";  break;
        case xor_op: s = "^";  break;
        case not_op: s = "!"; m = Prefix; break;
        case rshift_logical_op:break;
        case rshift_arithm_op:break;
        case lshift_op:break;
        case assign_op:break;
        case subscript_op:break;
        case alloca_op:break;
        case alloca_slot_op:break;
        case alloca_logical_op:break;
        case load_op: s = "*"; m = Prefix; break;
        case store_op:
            print(p, "\n*%s = %s;", emit_value(emitter, prim_op->operands.nodes[0]), emit_value(emitter, prim_op->operands.nodes[1]));
            break;
        case lea_op: {
            const char* acc = emit_value(emitter, prim_op->operands.nodes[0]);
            assert(acc);
            if (prim_op->operands.nodes[1])
                acc = format_string(emitter->arena, "&(%s[%s])", acc, emit_value(emitter, prim_op->operands.nodes[1]));
            const Type* t = extract_operand_type(prim_op->operands.nodes[0]->type);
            assert(t->tag == PtrType_TAG);
            t = t->payload.ptr_type.pointed_type;
            for (size_t i = 2; i < prim_op->operands.count; i++) {
                switch (is_type(t)) {
                    case ArrType_TAG: {
                        acc = format_string(emitter->arena, "&(%s[%s])", acc, emit_value(emitter, prim_op->operands.nodes[i]));
                        break;
                    }
                    case RecordType_TAG: error("TODO");
                    default: error("lea can't work on this");
                }
            }
            rhs = acc;
            break;
        }
        case select_op:break;
        case convert_op:break;
        case reinterpret_op: {
            rhs = format_string(emitter->arena, "(%s) %s", emit_type(emitter, prim_op->operands.nodes[0], NULL), emit_value(emitter, prim_op->operands.nodes[1]));
            break;
        }
        case extract_op:break;
        case extract_dynamic_op:break;
        case push_stack_op:break;
        case pop_stack_op:break;
        case push_stack_uniform_op:break;
        case pop_stack_uniform_op:break;
        case get_stack_pointer_op:break;
        case get_stack_pointer_uniform_op:break;
        case set_stack_pointer_op:break;
        case set_stack_pointer_uniform_op:break;
        case subgroup_elect_first_op:break;
        case subgroup_broadcast_first_op:break;
        case subgroup_active_mask_op:break;
        case subgroup_ballot_op:break;
        case subgroup_local_id_op:break;
        case empty_mask_op:break;
        case mask_is_thread_active_op:break;
        case debug_printf_op:break;
        case PRIMOPS_COUNT: assert(false); break;
    }

    if (outputs == NULL || outputs->count == 0)
        return;

    LARRAY(const char*, named, outputs->count);
    for (size_t i = 0; i < outputs->count; i++) {
        const Variable* var = &outputs->nodes[i]->payload.var;
        named[i] = format_string(emitter->arena, "%s_%d", var->name, var->id);
        insert_dict(const Node*, String, emitter->emitted, outputs->nodes[i], named[i]);
    }

    assert(outputs->count == 1);
    const Variable* var = &outputs->nodes[0]->payload.var;
    String decl = emit_type(emitter, outputs->nodes[0]->type, format_string(emitter->arena, "const %s_%d", var->name, var->id));

    if (s == NULL) {
        if (rhs)
            print(p, "\n%s = %s;", decl, rhs);
        else
            print(p, "\n%s; /* todo: implement %s */", decl, primop_names[prim_op->op]);
        return;
    }

    switch (m) {
        case Infix: print(p, "\n%s = %s %s %s;", decl, emit_value(emitter, prim_op->operands.nodes[0]), s, emit_value(emitter, prim_op->operands.nodes[1])); break;
        case Prefix: print(p, "\n%s = %s%s;", decl, s, emit_value(emitter, prim_op->operands.nodes[0])); break;
    }
}

static String emit_callee(Emitter* e, const Node* callee) {
    if (callee->tag == Function_TAG)
        return callee->payload.fn.name;
    else
        return emit_value(e, callee);
}

static void emit_call(Emitter* emitter, Printer* p, const Call* call, const Type* result_type, const Nodes* outputs) {
    Growy* g = new_growy();
    Printer* paramsp = open_growy_as_printer(g);
    for (size_t i = 0; i < call->args.count; i++) {
        print(paramsp, emit_value(emitter, call->args.nodes[i]));
        if (i + 1 < call->args.count)
            print(paramsp, ", ");
    }
    String params = printer_growy_unwrap(paramsp);

    if (outputs->count > 1) {
        declare_variables_helper(emitter, p, outputs);
        String named = unique_name(emitter->arena, "result");
        print(p, "\n%s = %s(%s);", emit_type(emitter, result_type, named), emit_callee(emitter, call->callee), params);
        emit_unpack_code(emitter, p, named, outputs);
    } else if (outputs->count == 1) {
        const Variable* var = &outputs->nodes[0]->payload.var;
        String named = format_string(emitter->arena, "%s_%d", var->name, var->id);
        print(p, "\n%s = %s(%s);", emit_type(emitter, var->type, named), emit_callee(emitter, call->callee), params);
        insert_dict(const Node*, String, emitter->emitted, outputs->nodes[0], named);
    } else {
        print(p, "\n%s(%s);", emit_callee(emitter, call->callee), params);
    }
    free(params);
}

static void emit_if(Emitter* emitter, Printer* p, const If* if_instr, const Nodes* outputs) {
    if (outputs->count > 0)
        print(p, "\n/* if yield values */");
    declare_variables_helper(emitter, p, outputs);

    Emitter sub_emiter = *emitter;
    sub_emiter.phis.selection = outputs;

    String true_block = emit_block_helper(&sub_emiter, if_instr->if_true, NULL);
    String false_block = if_instr->if_false ? emit_block_helper(&sub_emiter, if_instr->if_false, NULL) : NULL;
    print(p, "\nif (%s) %s", emit_value(emitter, if_instr->condition), true_block);
    if (false_block)
        print(p, " else %s", false_block);
    free(true_block);
    free(false_block);
}

static void emit_match(Emitter* emitter, Printer* p, const Match* match_instr, const Nodes* outputs) {
    if (outputs->count > 0)
        print(p, "\n/* match yield values */");
    declare_variables_helper(emitter, p, outputs);

    Emitter sub_emiter = *emitter;
    sub_emiter.phis.selection = outputs;

    print(p, "\nswitch (%s) {", emit_value(emitter, match_instr->inspect));
    indent(p);
    for (size_t i = 0; i < match_instr->cases.count; i++) {
        String case_body = emit_block_helper(&sub_emiter, match_instr->cases.nodes[i], NULL);
        print(p, "\ncase %s: %s\n", emit_value(emitter, match_instr->literals.nodes[i]), case_body);
        free(case_body);
    }
    if (match_instr->default_case) {
        String default_case_body = emit_block_helper(&sub_emiter, match_instr->default_case, NULL);
        print(p, "\ndefault: %s\n", default_case_body);
        free(default_case_body);
    }
    deindent(p);
    print(p, "\n}");
}

static void emit_loop(Emitter* emitter, Printer* p, const Loop* loop_instr, const Nodes* outputs) {
    if (loop_instr->params.count > 0)
        print(p, "\n/* loop parameters */");
    declare_variables_helper(emitter, p, &loop_instr->params);
    if (outputs->count > 0)
        print(p, "\n/* loop yield values */");
    declare_variables_helper(emitter, p, outputs);

    Emitter sub_emiter = *emitter;
    sub_emiter.phis.loop_continue = &loop_instr->params;
    sub_emiter.phis.loop_break = outputs;

    String body = emit_block_helper(&sub_emiter, loop_instr->body, NULL);
    print(p, "\nwhile(true) %s", body);
    free(body);
}

static void emit_instruction(Emitter* emitter, Printer* p, const Node* instruction) {
    assert(is_instruction(instruction));
    Nodes vars = nodes(emitter->arena, 0, NULL);

    if (instruction->tag == Let_TAG) {
        vars = instruction->payload.let.variables;
        instruction = instruction->payload.let.instruction;
    }

    switch (is_instruction(instruction)) {
        case Instruction_Let_TAG:
        case NotAnInstruction: assert(false);
        case Instruction_PrimOp_TAG: emit_primop(emitter, p, &instruction->payload.prim_op, &vars);     break;
        case Instruction_Call_TAG:   emit_call  (emitter, p, &instruction->payload.call_instr, instruction->type, &vars);  break;
        case Instruction_If_TAG:     emit_if    (emitter, p, &instruction->payload.if_instr, &vars);    break;
        case Instruction_Match_TAG:  emit_match (emitter, p, &instruction->payload.match_instr, &vars); break;
        case Instruction_Loop_TAG:   emit_loop  (emitter, p, &instruction->payload.loop_instr, &vars);  break;
    }
}

static void emit_terminator(Emitter* emitter, Printer* p, const Node* terminator) {
    switch (is_terminator(terminator)) {
        case NotATerminator: assert(false);
        case Terminator_Join_TAG:
        case Terminator_Callc_TAG:  error("this must be lowered away!");
        case Terminator_Branch_TAG: break;
        case Terminator_Return_TAG: {
            Nodes args = terminator->payload.fn_ret.values;
            if (args.count == 0) {
                print(p, "\nreturn;");
            } else if (args.count == 1) {
                print(p, "\nreturn %s;", emit_value(emitter, args.nodes[0]));
            } else {
                String packed = unique_name(emitter->arena, "pack_return");
                emit_pack_code(emitter, p, &args, packed);
                print(p, "\nreturn %s;", packed);
            }
            break;
        }
        case Terminator_MergeConstruct_TAG: {
            Nodes args = terminator->payload.merge_construct.args;
            const Nodes* phis;
            switch (terminator->payload.merge_construct.construct) {
                case Selection: phis = emitter->phis.selection; break;
                case Continue:  phis = emitter->phis.loop_continue; break;
                case Break:     phis = emitter->phis.loop_break; break;
            }
            assert(phis && phis->count == args.count);
            for (size_t i = 0; i < phis->count; i++) {
                print(p, "\n%s = %s;", emit_value(emitter, phis->nodes[i]), emit_value(emitter, args.nodes[i]));
            }

            switch (terminator->payload.merge_construct.construct) {
                case Selection: break;
                case Continue:  print(p, "\ncontinue;"); break;
                case Break:     print(p, "\nbreak;"); break;
            }
            break;
        }
        case Terminator_Unreachable_TAG: {
            assert(emitter->config.dialect == C);
            print(p, "\n__builtin_unreachable();");
            break;
        }
    }
}

static void emit_block(Emitter* emitter, Printer* p, const Node* block) {
    assert(block && block->tag == Block_TAG);
    print(p, "{");
    indent(p);

    for (size_t i = 0; i < block->payload.block.instructions.count; i++) {
        emit_instruction(emitter, p, block->payload.block.instructions.nodes[i]);
    }
    
    emit_terminator(emitter, p, block->payload.block.terminator);

    deindent(p);
    print(p, "\n}");
}

static String emit_block_helper(Emitter* emitter, const Node* block, const Nodes* bbs) {
    Growy* g = new_growy();
    Printer* p = open_growy_as_printer(g);
    emit_block(emitter, p, block);
    growy_append_bytes(g, 1, (char[]) { 0 });

    if (bbs && bbs->count > 0) {
        assert(emitter->config.dialect != GLSL);
        error("TODO");
    }

    return printer_growy_unwrap(p);
}

static String emit_decl(Emitter* emitter, const Node* decl) {
    assert(is_declaration(decl->tag));

    String* found = find_value_dict(const Node*, String, emitter->emitted, decl);
    if (found)
        return *found;

    const char* name = get_decl_name(decl);
    const Type* decl_type = decl->type;
    const char* decl_center = name;
    const char* emit_as = NULL;

    switch (decl->tag) {
        case GlobalVariable_TAG: {
            decl_type = decl->payload.global_variable.type;
            // users of the global variable are actually using its address
            emit_as = format_string(emitter->arena, "(&%s)", name);
            insert_dict(const Node*, String, emitter->emitted, decl, emit_as);
            if (decl->payload.global_variable.init)
                print(emitter->fn_defs, "\n%s = %s;", emit_type(emitter, decl_type, decl_center), emit_value(emitter, decl->payload.global_variable.init));
            break;
        }
        case Function_TAG: {
            emit_as = name;
            insert_dict(const Node*, String, emitter->emitted, decl, emit_as);
            if (decl->payload.fn.block) {
                for (size_t i = 0; i < decl->payload.fn.params.count; i++) {
                    const char* param_name = format_string(emitter->arena, "%s_%d", decl->payload.fn.params.nodes[i]->payload.var.name, decl->payload.fn.params.nodes[i]->payload.var.id);
                    insert_dict(const Node*, String, emitter->emitted, decl->payload.fn.params.nodes[i], param_name);
                }

                String fn_body = emit_block_helper(emitter, decl->payload.fn.block, NULL);
                print(emitter->fn_defs, "\n%s %s", emit_fn_head(emitter, decl), fn_body);
                free(fn_body);
            }
            break;
        }
        case Constant_TAG: {
            emit_as = name;
            insert_dict(const Node*, String, emitter->emitted, decl, emit_as);
            decl_center = format_string(emitter->arena, "const %s", decl_center);
            print(emitter->fn_defs, "\n%s = %s;", emit_type(emitter, decl->type, decl_center), emit_value(emitter, decl->payload.constant.value));
            break;
        }
        default: error("not a decl");
    }

    String declaration = emit_type(emitter, decl_type, decl_center);
    print(emitter->fn_decls, "\n%s;", declaration);
    return emit_as;
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
        .arena = arena,
        .type_decls = open_growy_as_printer(type_decls_g),
        .fn_decls = open_growy_as_printer(fn_decls_g),
        .fn_defs = open_growy_as_printer(fn_defs_g),
        .emitted = new_dict(Node*, String, (HashFn) hash_node, (CmpFn) compare_node)
    };

    assert(root_node->tag == Root_TAG);
    Nodes decls = root_node->payload.root.declarations;
    for (size_t i = 0; i < decls.count; i++)
        emit_decl(&emitter, decls.nodes[i]);

    destroy_printer(emitter.type_decls);
    destroy_printer(emitter.fn_decls);
    destroy_printer(emitter.fn_defs);

    Growy* final = new_growy();
    Printer* finalp = open_growy_as_printer(final);

    print(finalp, "/* file generated by shady */\n");

    if (emitter.config.dialect == C) {
        print(finalp, "\n#include <stdbool.h>");
        print(finalp, "\n#include <stdint.h>");
        print(finalp, "\n#include <stddef.h>");
    }

    print(finalp, "\n/* types: */\n");
    growy_append_bytes(final, growy_size(type_decls_g), growy_data(type_decls_g));

    print(finalp, "\n/* declarations: */\n");
    growy_append_bytes(final, growy_size(fn_decls_g), growy_data(fn_decls_g));

    print(finalp, "\n/* definitions: */\n");
    growy_append_bytes(final, growy_size(fn_defs_g), growy_data(fn_defs_g));

    print(finalp, "\n");

    growy_destroy(type_decls_g);
    growy_destroy(fn_decls_g);
    growy_destroy(fn_defs_g);

    destroy_dict(emitter.emitted);

    *output_size = growy_size(final);
    *output = growy_deconstruct(final);
    destroy_printer(finalp);
}
