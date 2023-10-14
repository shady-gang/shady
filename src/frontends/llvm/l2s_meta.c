#include "l2s_private.h"

#include "portability.h"
#include "log.h"

#include "llvm-c/DebugInfo.h"

static Nodes convert_mdnode_operands(Parser* p, LLVMValueRef mdnode) {
    IrArena* a = get_module_arena(p->dst);
    assert(LLVMIsAMDNode(mdnode));

    unsigned count = LLVMGetMDNodeNumOperands(mdnode);
    LARRAY(LLVMValueRef, ops, count);
    LLVMGetMDNodeOperands(mdnode, ops);

    LARRAY(const Node*, cops, count);
    for (size_t i = 0; i < count; i++)
        cops[i] = ops[i] ? convert_value(p, ops[i]) : string_lit_helper(a, "null");
    Nodes args = nodes(a, count, cops);
    return args;
}

static const Node* convert_named_tuple_metadata(Parser* p, LLVMValueRef v, String name) {
    IrArena* a = get_module_arena(p->dst);
    Nodes args = convert_mdnode_operands(p, v);
    args = prepend_nodes(a, args, string_lit_helper(a, name));
    return tuple(a, args);
}

const Node* convert_metadata(Parser* p, LLVMMetadataRef meta) {
    IrArena* a = get_module_arena(p->dst);
    LLVMMetadataKind kind = LLVMGetMetadataKind(meta);
    LLVMValueRef v = LLVMMetadataAsValue(p->ctx, meta);

    switch (kind) {
        case LLVMMDStringMetadataKind: {
            unsigned l;
            String name = LLVMGetMDString(v, &l);
            return string_lit_helper(a, name);
        }
        case LLVMConstantAsMetadataMetadataKind:
        case LLVMLocalAsMetadataMetadataKind: {
            Nodes ops = convert_mdnode_operands(p, v);
            assert(ops.count == 1);
            return first(ops);
        }
        case LLVMDistinctMDOperandPlaceholderMetadataKind: goto default_;
        case LLVMMDTupleMetadataKind: return tuple(a, convert_mdnode_operands(p, v));

        case LLVMDILocationMetadataKind:                 return convert_named_tuple_metadata(p, v, "DILocation");
        case LLVMDIExpressionMetadataKind:               return convert_named_tuple_metadata(p, v, "DIExpression");
        case LLVMDIGlobalVariableExpressionMetadataKind: return convert_named_tuple_metadata(p, v, "DIGlobalVariableExpression");
        case LLVMGenericDINodeMetadataKind:              return convert_named_tuple_metadata(p, v, "GenericDINode");
        case LLVMDISubrangeMetadataKind:                 return convert_named_tuple_metadata(p, v, "DISubrange");
        case LLVMDIEnumeratorMetadataKind:               return convert_named_tuple_metadata(p, v, "DIEnumerator");
        case LLVMDIBasicTypeMetadataKind:                return convert_named_tuple_metadata(p, v, "DIBasicType");
        case LLVMDIDerivedTypeMetadataKind:              return convert_named_tuple_metadata(p, v, "DIDerivedType");
        case LLVMDICompositeTypeMetadataKind:            return convert_named_tuple_metadata(p, v, "DICompositeType");
        case LLVMDISubroutineTypeMetadataKind:           return convert_named_tuple_metadata(p, v, "DISubroutineType");
        case LLVMDIFileMetadataKind:                     return convert_named_tuple_metadata(p, v, "DIFile");
        case LLVMDICompileUnitMetadataKind:              return convert_named_tuple_metadata(p, v, "DICompileUnit");
        case LLVMDISubprogramMetadataKind:               return convert_named_tuple_metadata(p, v, "DiSubprogram");
        case LLVMDILexicalBlockMetadataKind:             return convert_named_tuple_metadata(p, v, "DILexicalBlock");
        case LLVMDILexicalBlockFileMetadataKind:         return convert_named_tuple_metadata(p, v, "DILexicalBlockFile");
        case LLVMDINamespaceMetadataKind:                return convert_named_tuple_metadata(p, v, "DINamespace");
        case LLVMDIModuleMetadataKind:                   return convert_named_tuple_metadata(p, v, "DIModule");
        case LLVMDITemplateTypeParameterMetadataKind:    return convert_named_tuple_metadata(p, v, "DITemplateTypeParameter");
        case LLVMDITemplateValueParameterMetadataKind:   return convert_named_tuple_metadata(p, v, "DITemplateValueParameter");
        case LLVMDIGlobalVariableMetadataKind:           return convert_named_tuple_metadata(p, v, "DIGlobalVariable");
        case LLVMDILocalVariableMetadataKind:            return convert_named_tuple_metadata(p, v, "DILocalVariable");
        case LLVMDILabelMetadataKind:                    return convert_named_tuple_metadata(p, v, "DILabelMetadata");
        case LLVMDIObjCPropertyMetadataKind:             return convert_named_tuple_metadata(p, v, "DIObjCProperty");
        case LLVMDIImportedEntityMetadataKind:           return convert_named_tuple_metadata(p, v, "DIImportedEntity");
        case LLVMDIMacroMetadataKind:                    return convert_named_tuple_metadata(p, v, "DIMacroMetadata");
        case LLVMDIMacroFileMetadataKind:                return convert_named_tuple_metadata(p, v, "DIMacroFile");
        case LLVMDICommonBlockMetadataKind:              return convert_named_tuple_metadata(p, v, "DICommonBlock");
        case LLVMDIStringTypeMetadataKind:               return convert_named_tuple_metadata(p, v, "DIStringType");
        case LLVMDIGenericSubrangeMetadataKind:          return convert_named_tuple_metadata(p, v, "DIGenericSubrange");
        case LLVMDIArgListMetadataKind:                  return convert_named_tuple_metadata(p, v, "DIArgList");
        default: default_:
            error_print("Unknown metadata kind %d for ", kind);
            LLVMDumpValue(v);
            error_print(".\n");
            error_die();
    }
}
