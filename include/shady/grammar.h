#ifndef SHADY_IR_H
#error "do not include this file by itself, include shady/ir.h instead"
#endif

/// In this file, we establish the grammar of the Shady language
/// We make intense use of X-macros to define an ADT (Abstract Data Type)
/// The NODES() macro lists all the possible alternatives of the `Node` sum type
// Each node is defined as such:
// N(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name)

#define TYPE_NODES(N) \
N(1, 0, 0, MaskType, mask_type) \
N(1, 1, 1, JoinPointType, join_point_type) \
N(1, 0, 0, NoRet, noret_type) \
N(1, 0, 1, Int, int_type) \
N(1, 0, 1, Float, float_type) \
N(1, 0, 0, Bool, bool_type) \
N(1, 1, 1, RecordType, record_type) \
N(1, 0, 1, FnType, fn_type) \
N(1, 0, 1, BBType, bb_type) \
N(1, 0, 1, LamType, lam_type) \
N(1, 1, 1, PtrType, ptr_type) \
N(1, 1, 1, QualifiedType, qualified_type) \
N(1, 1, 1, ArrType, arr_type) \
N(1, 1, 1, PackType, pack_type) \
N(1, 0, 1, TypeDeclRef, type_decl_ref) \
N(1, 0, 1, ImageType, image_type) \
N(1, 0, 0, SamplerType, sampler_type) \
N(1, 0, 1, CombinedImageSamplerType, combined_image_sampler_type) \

#define VALUE_NODES(N) \
N(0, 1, 1, Variable, var) \
N(1, 0, 1, ConstrainedValue, constrained) \
N(1, 1, 1, UntypedNumber, untyped_number) \
N(1, 1, 1, IntLiteral, int_literal) \
N(1, 1, 1, FloatLiteral, float_literal) \
N(1, 1, 0, True, true_lit) \
N(1, 1, 0, False, false_lit) \
N(1, 1, 1, StringLiteral, string_lit) \
N(1, 1, 1, NullPtr, null_ptr) \
N(1, 1, 1, Composite, composite) \
N(1, 1, 1, Fill, fill) \
N(1, 1, 1, Undef, undef) \
N(1, 1, 1, FnAddr, fn_addr) \
N(1, 1, 1, RefDecl, ref_decl) \
N(1, 1, 1, AntiQuote, anti_quote) \

#define INSTRUCTION_NODES(N) \
N(1, 1, 1, Call, call) \
N(1, 1, 1, PrimOp, prim_op)  \
N(1, 1, 1, If, if_instr) \
N(1, 1, 1, Match, match_instr) \
N(1, 1, 1, Loop, loop_instr) \
N(1, 1, 1, Control, control) \
N(1, 1, 1, Block, block) \
N(1, 1, 1, Comment, comment) \

#define TERMINATOR_NODES(N) \
N(0, 1, 1, Let, let) \
N(0, 0, 1, LetMut, let_mut) \
N(1, 1, 1, TailCall, tail_call) \
N(1, 1, 1, Jump, jump) \
N(1, 1, 1, Branch, branch) \
N(1, 1, 1, Switch, br_switch) \
N(1, 1, 1, Join, join) \
N(1, 1, 1, MergeContinue, merge_continue) \
N(1, 1, 1, MergeBreak, merge_break) \
N(1, 1, 1, Yield, yield) \
N(1, 1, 1, Return, fn_ret) \
N(1, 1, 0, Unreachable, unreachable) \

#define DECL_NODES(N) \
N(0, 1, 1, Function, fun) \
N(0, 1, 1, Constant, constant) \
N(0, 1, 1, GlobalVariable, global_variable) \
N(0, 0, 1, NominalType, nom_type) \

#define ANNOTATION_NODES(N) \
N(1, 0, 1, Annotation, annotation) \
N(1, 0, 1, AnnotationValue, annotation_value) \
N(1, 0, 1, AnnotationValues, annotation_values) \
N(1, 0, 1, AnnotationCompound, annotations_compound) \

#define NODES(N) \
TYPE_NODES(N) \
VALUE_NODES(N) \
INSTRUCTION_NODES(N) \
TERMINATOR_NODES(N) \
DECL_NODES(N) \
ANNOTATION_NODES(N) \
N(0, 1, 1, AnonLambda, anon_lam) \
N(0, 1, 1, BasicBlock, basic_block) \
N(1, 0, 1, Unbound, unbound) \
N(1, 0, 1, UnboundBBs, unbound_bbs)

/// We declare the payloads of our nodes using this special
/// X-macro pattern in order to be able to 'reflect' them
/// for generating processing code using yet more x-macros
///
/// It's x-macros all the way down.

//////////////////////////////// Types ////////////////////////////////

typedef enum DivergenceQualifier_ {
    Unknown,
    Uniform,
    Varying
} DivergenceQualifier;

typedef struct QualifiedType_ QualifiedType;
#define QualifiedType_Fields(MkField) \
MkField(1, POD, bool, is_uniform) \
MkField(1, TYPE, const Type*, type)

typedef enum {
    NotSpecial,
    /// for instructions with multiple yield values. Must be deconstructed by a let, cannot appear anywhere else
    MultipleReturn,
    /// Gets the 'Block' SPIR-V annotation, needed for UBO/SSBO variables
    DecorateBlock
} RecordSpecialFlag;

typedef struct RecordType_ RecordType;
#define RecordType_Fields(MkField) \
MkField(1, TYPES, Nodes, members) \
MkField(1, STRINGS, Strings, names) \
MkField(1, POD, RecordSpecialFlag, special)

typedef struct JoinPointType_ JoinPointType;
#define JoinPointType_Fields(MkField) \
MkField(1, TYPES, Nodes, yield_types)

typedef struct FnType_ FnType;
#define FnType_Fields(MkField) \
MkField(1, TYPES, Nodes, param_types) \
MkField(1, TYPES, Nodes, return_types)

typedef struct BBType_ BBType;
#define BBType_Fields(MkField) \
MkField(1, TYPES, Nodes, param_types)

typedef struct LamType_ LamType;
#define LamType_Fields(MkField) \
MkField(1, TYPES, Nodes, param_types)

typedef struct ImageType_ ImageType;
#define ImageType_Fields(MkField) \
MkField(1, TYPE, const Type*, sampled_type) \
MkField(1, POD, uint32_t, dim) \
MkField(1, POD, uint32_t, depth) \
MkField(1, POD, bool, onion) \
MkField(1, POD, bool, multisample) \
MkField(1, POD, uint32_t, sampled) \

typedef struct CombinedImageSamplerType_ CombinedImageSamplerType;
#define CombinedImageSamplerType_Fields(MkField) \
MkField(1, TYPE, const Type*, image_type)

typedef struct PtrType_ PtrType;
#define PtrType_Fields(MkField) \
MkField(1, POD, AddressSpace, address_space) \
MkField(1, TYPE, const Type*, pointed_type)

typedef struct ArrType_ ArrType;
#define ArrType_Fields(MkField) \
MkField(1, TYPE, const Type*, element_type) \
MkField(1, VALUE, const Node*, size)

typedef enum {
    IntTy8,
    IntTy16,
    IntTy32,
    IntTy64,
} IntSizes;

enum {
    IntSizeMin = IntTy8,
    IntSizeMax = IntTy64,
};

typedef struct Int_ Int;
#define Int_Fields(MkField) \
MkField(1, POD, IntSizes, width) \
MkField(1, POD, bool, is_signed)

typedef enum {
    FloatTy16,
    FloatTy32,
    FloatTy64
} FloatSizes;

typedef struct Float_ Float;
#define Float_Fields(MkField) \
MkField(1, POD, FloatSizes, width)

typedef struct PackType_ PackType;
#define PackType_Fields(MkField) \
MkField(1, TYPE, const Type*, element_type) \
MkField(1, POD, int, width)

typedef struct TypeDeclRef_ TypeDeclRef;
#define TypeDeclRef_Fields(MkField) \
MkField(1, DECL, const Node*,  decl)

//////////////////////////////// Values ////////////////////////////////

typedef struct Variable_ Variable;
#define Variable_Fields(MkField) \
MkField(1, TYPE, const Type*, type) \
MkField(1, POD, VarId, id) \
MkField(1, STRING, String, name) \
MkField(0, SCRATCH, const Node*, abs) \
MkField(0, SCRATCH, unsigned, pindex)

typedef struct ConstrainedValue_ ConstrainedValue;
#define ConstrainedValue_Fields(MkField) \
MkField(1, TYPE, const Type*, type) \
MkField(1, VALUE, const Node*, value) \

typedef struct UntypedNumber_ UntypedNumber;
#define UntypedNumber_Fields(MkField) \
MkField(1, POD, String, plaintext)

typedef struct IntLiteral_ IntLiteral;
#define IntLiteral_Fields(MkField) \
MkField(1, POD, IntSizes, width) \
MkField(1, POD, bool, is_signed) \
MkField(1, POD, uint64_t, value)

/// C lacks sized float types, so let's just store the raw bits for them.
typedef struct FloatLiteral_ FloatLiteral;
#define FloatLiteral_Fields(MkField) \
MkField(1, POD, FloatSizes, width) \
MkField(1, POD, uint64_t, value)

typedef struct StringLiteral_ StringLiteral;
#define StringLiteral_Fields(MkField) \
MkField(1, STRING, String, string)

typedef struct NullPtr_ NullPtr;
#define NullPtr_Fields(MkField) \
MkField(1, TYPE, const Type*, ptr_type)

/// A value made out of more values.
/// Re-ordering values does not count as a computation here !
typedef struct Composite_ Composite;
#define Composite_Fields(MkField) \
MkField(1, TYPE, const Type*, type) \
MkField(1, VALUES, Nodes, contents)

typedef struct Fill_ Fill;
#define Fill_Fields(MkField) \
MkField(1, TYPE, const Type*, type) \
MkField(1, VALUE, const Node*, value)

typedef struct Undef_ Undef;
#define Undef_Fields(MkField) \
MkField(1, TYPE, const Type*, type)

/// References either a global (yielding a pointer to it), or a constant (yielding a value of the type itself)
/// Declarations are not values themselves, this node is required to "convert" them.
typedef struct RefDecl_ RefDecl;
#define RefDecl_Fields(MkField) \
MkField(1, DECL, const Node*, decl)

/// Like RefDecl but for functions, it yields a _function pointer_ !
typedef struct FnAddr_ FnAddr;
#define FnAddr_Fields(MkField) \
MkField(1, DECL, const Node*, fn)

/// Like RefDecl but for functions, it yields a _function pointer_ !
typedef struct AntiQuote_ AntiQuote;
#define AntiQuote_Fields(MkField) \
MkField(1, INSTRUCTION, const Node*, instruction)

//////////////////////////////// Instructions ////////////////////////////////

typedef struct PrimOp_ PrimOp;
#define PrimOp_Fields(MkField) \
MkField(1, POD, Op, op) \
MkField(1, TYPES, Nodes, type_arguments) \
MkField(1, VALUES, Nodes, operands)

typedef struct Call_ Call;
#define Call_Fields(MkField) \
MkField(1, VALUE, const Node*, callee) \
MkField(1, VALUES, Nodes, args)

// Those things are "meta" instructions, they contain other instructions.
// they map to SPIR-V structured control flow constructs directly
// they don't need merge blocks because they are instructions and so that is taken care of by the containing node

/// Structured "if" construct
typedef struct If_ If;
#define If_Fields(MkField) \
MkField(1, TYPES, Nodes, yield_types) \
MkField(1, VALUE, const Node*, condition) \
MkField(1, ANON_LAMBDA, const Node*, if_true) \
MkField(1, ANON_LAMBDA, const Node*, if_false)

/// Structured "match" construct
typedef struct Match_ Match;
#define Match_Fields(MkField) \
MkField(1, TYPES, Nodes, yield_types) \
MkField(1, VALUE, const Node*, inspect) \
MkField(1, VALUES, Nodes, literals) \
MkField(1, ANON_LAMBDAS, Nodes, cases) \
MkField(1, ANON_LAMBDA, const Node*, default_case)

/// Structured "block" construct
/// used as a helper block to insert multiple instructions in place of one
typedef struct Block_ Block;
#define Block_Fields(MkField) \
MkField(1, TYPES, Nodes, yield_types) \
MkField(1, ANON_LAMBDA, const Node*, inside)

/// Structured "loop" construct
typedef struct Loop_ Loop;
#define Loop_Fields(MkField) \
MkField(1, TYPES, Nodes, yield_types) \
MkField(1, ANON_LAMBDA, const Node*, body) \
MkField(1, VALUES, Nodes, initial_args)

/// Structured "control" construct
typedef struct Control_ Control;
#define Control_Fields(MkField) \
MkField(1, TYPES, Nodes, yield_types) \
MkField(1, ANON_LAMBDA, const Node*, inside)

/// Structured "control" construct
typedef struct Comment_ Comment;
#define Comment_Fields(MkField) \
MkField(1, STRING, String, string) \

//////////////////////////////// Terminators ////////////////////////////////

typedef struct Let_ Let;
#define Let_Fields(MkField) \
MkField(1, INSTRUCTION, const Node*, instruction) \
MkField(1, ANON_LAMBDA, const Node*, tail)

typedef struct LetMut_ LetMut;
#define LetMut_Fields(MkField) \
MkField(1, INSTRUCTION, const Node*, instruction) \
MkField(1, ANON_LAMBDA, const Node*, tail)

typedef struct Jump_ Jump;
#define Jump_Fields(MkField) \
MkField(1, BASIC_BLOCK, const Node*, target) \
MkField(1, VALUES, Nodes, args)

/// A branch. Branches can cause divergence, but they can never cause re-convergence.
typedef struct Branch_ Branch;
#define Branch_Fields(MkField) \
MkField(1, VALUE, const Node*, branch_condition) \
MkField(1, TERMINATOR, const Node*, true_jump) \
MkField(1, TERMINATOR, const Node*, false_jump) \

typedef struct Switch_ Switch;
#define Switch_Fields(MkField) \
MkField(1, VALUE, const Node*, switch_value) \
MkField(1, VALUES, Nodes, case_values) \
MkField(1, TERMINATORS, Nodes, case_jumps) \
MkField(1, TERMINATOR, const Node*, default_jump)

/// Join nodes are used to undo the divergence caused by branches. At join nodes, an explicit mask is used to force a number of divergent execution paths to resume.
/// If @p is_indirect is set, the target must be a function pointer. Otherwise, the target must be a function directly.
/// @p join_at _must_ be uniform.
typedef struct Join_ Join;
#define Join_Fields(MkField) \
MkField(1, VALUE, const Node*, join_point) \
MkField(1, VALUES, Nodes, args)

typedef struct Return_ Return;
#define Return_Fields(MkField) \
MkField(1, DECL, const Node*, fn) \
MkField(1, VALUES, Nodes, args)

typedef struct TailCall_ TailCall;
#define TailCall_Fields(MkField) \
MkField(1, VALUE, const Node*, target) \
MkField(1, VALUES, Nodes, args)

// These terminators are used in conjunction with structured constructs, they are used inside their bodies to yield a value
// Using those terminators outside an appropriate structured construct is illegal

typedef struct Yield_ Yield;
#define Yield_Fields(MkField) \
MkField(1, VALUES, Nodes, args)

typedef struct MergeContinue_ MergeContinue;
#define MergeContinue_Fields(MkField) \
MkField(1, VALUES, Nodes, args)

typedef struct MergeBreak_ MergeBreak;
#define MergeBreak_Fields(MkField) \
MkField(1, VALUES, Nodes, args)

//////////////////////////////// Decls ////////////////////////////////

/// Populated by the parser for the bind pass, should be empty at all other times after that
/// (use the Scope analysis to figure out the real scope of a function)
typedef Nodes ChildrenBlocks;

typedef struct Function_ Function;
#define Function_Fields(MkField) \
MkField(1, VARIABLES, Nodes, params) \
MkField(1, TERMINATOR, const Node*, body) \
MkField(0, POD, Module*, module) \
MkField(1, STRING, String, name) \
MkField(1, TYPES, Nodes, return_types) \
MkField(1, ANNOTATIONS, Nodes, annotations)

#define EXECUTION_MODELS(EM) \
EM(Compute,  1) \
EM(Fragment, 0) \
EM(Vertex,   0) \

typedef enum {
    EmNone,
#define EM(name, _) Em##name,
EXECUTION_MODELS(EM)
#undef EM
} ExecutionModel;

ExecutionModel execution_model_from_string(const char*);

typedef struct Constant_ Constant;
#define Constant_Fields(MkField) \
MkField(1, ANNOTATIONS, Nodes, annotations) \
MkField(0, POD, Module*, module) \
MkField(1, STRING, String, name) \
MkField(1, TYPE, const Type*, type_hint) \
MkField(0, VALUE, const Node*, value)

typedef struct GlobalVariable_ GlobalVariable;
#define GlobalVariable_Fields(MkField) \
MkField(1, ANNOTATIONS, Nodes, annotations) \
MkField(1, TYPE, const Type*, type) \
MkField(0, POD, Module*, module) \
MkField(1, STRING, String, name) \
MkField(1, POD, AddressSpace, address_space) \
MkField(0, VALUE, const Node*, init)

typedef struct NominalType_ NominalType;
#define NominalType_Fields(MkField) \
MkField(1, ANNOTATIONS, Nodes, annotations) \
MkField(0, POD, Module*, module) \
MkField(1, STRING, String, name) \
MkField(1, TYPE, const Type*, body)

//////////////////////////////// Misc ////////////////////////////////

/// A named abstraction that lives inside a function and can be jumped to
typedef struct BasicBlock_ BasicBlock;
#define BasicBlock_Fields(MkField) \
MkField(1, VARIABLES, Nodes, params) \
MkField(1, TERMINATOR, const Node*, body) \
MkField(1, DECL, const Node*, fn) \
MkField(1, STRING, String, name)

/// An unnamed abstraction that lives inside a function, and can be used as part of various control-flow constructs
/// Most notably, the tails of standard `let` nodes
typedef struct AnonLambda_ AnonLambda;
#define AnonLambda_Fields(MkField) \
MkField(1, VARIABLES, Nodes, params) \
MkField(1, TERMINATOR, const Node*, body) \
MkField(0, SCRATCH, const Node*, structured_construct) \

/// Unbound identifier, obtained by parsing a file
typedef struct Unbound_ Unbound;
#define Unbound_Fields(MkField) \
MkField(1, POD, String, name)

/// A node together with unbound basic blocks it dominates, obtained by parsing a file
typedef struct UnboundBBs_ UnboundBBs;
#define UnboundBBs_Fields(MkField) \
MkField(1, TERMINATOR, const Node*, body) \
MkField(1, BASIC_BLOCKS, ChildrenBlocks, children_blocks)

typedef struct Annotation_ Annotation;
#define Annotation_Fields(MkField) \
MkField(1, STRING, String, name) \

typedef struct AnnotationValue_ AnnotationValue;
#define AnnotationValue_Fields(MkField) \
MkField(1, STRING, String, name) \
MkField(1, VALUE, const Node*, value)

typedef struct AnnotationValues_ AnnotationValues;
#define AnnotationValues_Fields(MkField) \
MkField(1, STRING, String, name) \
MkField(1, VALUES, Nodes, values)

typedef struct AnnotationCompound_ AnnotationCompound;
#define AnnotationCompound_Fields(MkField) \
MkField(1, STRING, String, name) \
MkField(1, ANNOTATIONS, Nodes, entries)

// This macro is used to define what the 'field types' column in the _Fields macros before mean.
// These 'field types' are relevant for working with the grammar, they help distinguish values,
// types, instructions etc. from each other, and enable writing special rules for each category
#define GRAMMAR_FIELD_TYPES(FT) \
FT(0, POD)                      \
FT(0, STRING)                   \
FT(0, STRINGS)                  \
FT(1, TYPE)                     \
FT(1, TYPES)                    \
FT(1, VALUE)                    \
FT(1, VALUES)                   \
FT(1, INSTRUCTION)              \
FT(1, TERMINATOR)               \
FT(1, DECL)                     \
FT(1, ANON_LAMBDA)              \
FT(1, ANON_LAMBDAS)             \
FT(1, BASIC_BLOCK)              \
FT(1, BASIC_BLOCKS)             \

//////////////////////////////// Extracted definitions ////////////////////////////////

#include "generated_grammar.h"

extern const char* node_tags[];
extern const bool node_type_has_payload[];

//////////////////////////////// Node categories ////////////////////////////////

typedef enum {
    NotAType = 0,
#define X(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) Type_##struct_name##_TAG = struct_name##_TAG,
TYPE_NODES(X)
#undef X
} TypeTag;

TypeTag is_type(const Node*);

typedef enum {
    NotAValue = 0,
#define X(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) Value_##struct_name##_TAG = struct_name##_TAG,
VALUE_NODES(X)
#undef X
} ValueTag;

ValueTag is_value(const Node*);

typedef enum {
    NotATerminator = 0,
#define X(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) Terminator_##struct_name##_TAG = struct_name##_TAG,
TERMINATOR_NODES(X)
#undef X
} TerminatorTag;

TerminatorTag is_terminator(const Node*);

typedef enum {
    NotAnInstruction = 0,
#define X(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) Instruction_##struct_name##_TAG = struct_name##_TAG,
INSTRUCTION_NODES(X)
#undef X
} InstructionTag;

InstructionTag is_instruction(const Node*);

typedef enum {
    NotADecl = 0,
#define X(autogen_ctor, has_typing_fn, has_payload, struct_name, short_name) Decl_##struct_name##_TAG = struct_name##_TAG,
DECL_NODES(X)
#undef X
} DeclTag;

DeclTag is_declaration(const Node*);

inline static bool is_nominal(const Node* node) {
    NodeTag tag = node->tag;
    if (node->tag == PrimOp_TAG && has_primop_got_side_effects(node->payload.prim_op.op))
        return true;
    return tag == Function_TAG || tag == BasicBlock_TAG || tag == Constant_TAG || tag == Variable_TAG || tag == GlobalVariable_TAG || tag == NominalType_TAG || tag == AnonLambda_TAG;
}

inline static bool is_anonymous_lambda(const Node* node) { return node->tag == AnonLambda_TAG; }
inline static bool is_basic_block(const Node* node) { return node->tag == BasicBlock_TAG; }
inline static bool is_function(const Node* node) { return node->tag == Function_TAG; }
