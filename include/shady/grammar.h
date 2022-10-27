typedef enum {
    POD,
    VALUE,
    TYPE,
} NodeFieldType;

// We declare the fields of our nodes using this special
// X-macro pattern in order to be able to 'reflect' them
// for generating processing code using yet more x-macros
//
// It's x-macros all the way down.

#define QualifiedType_Fields(MkField) \
MkField(1, POD, bool, is_uniform) \
MkField(1, TYPE, const Type*, type)

#define RecordType_Fields(MkField) \
MkField(1, TYPES, Nodes, members) \
MkField(1, STRINGS, Strings, names) \
MkField(1, POD, RecordSpecialFlag, special)

#define JoinPointType_Fields(MkField) \
MkField(1, TYPES, Nodes, yield_types)

#define FnType_Fields(MkField) \
MkField(1, POD, FnTier, tier) \
MkField(1, TYPES, Nodes, param_types) \
MkField(1, TYPES, Nodes, return_types)

#define PtrType_Fields(MkField) \
MkField(1, POD, AddressSpace, address_space) \
MkField(1, TYPE, const Type*, pointed_type)

#define ArrType_Fields(MkField) \
MkField(1, TYPE, const Type*, element_type) \
MkField(1, VALUE, const Node*, size)

#define Int_Fields(MkField) \
MkField(1, POD, IntSizes, width)

#define PackType_Fields(MkField) \
MkField(1, TYPE, const Type*, element_type) \
MkField(1, POD, int, width)

#define NominalType_Fields(MkField) \
MkField(1, STRING, String, name) \
MkField(1, TYPE, const Type*, body)

#define Variable_Fields(MkField) \
MkField(1, TYPE, const Type*, type) \
MkField(1, POD, VarId, id) \
MkField(1, STRING, String, name) \
MkField(0, INSTRUCTION, const Node*, instruction) \
MkField(0, POD, unsigned, output)

#define Unbound_Fields(MkField) \
MkField(1, POD, String, name)

#define UntypedNumber_Fields(MkField) \
MkField(1, POD, String, plaintext)

typedef union {
    int64_t  i64;
    int32_t  i32;
    int16_t  i16;
    int8_t    i8;
    uint64_t u64;
    uint32_t u32;
    uint16_t u16;
    uint8_t   u8;
} IntLiteralValue;

#define IntLiteral_Fields(MkField) \
MkField(1, POD, IntSizes, width) \
MkField(1, POD, IntLiteralValue, value)

#define StringLiteral_Fields(MkField) \
MkField(1, STRING, String, string)

#define ArrayLiteral_Fields(MkField) \
MkField(1, TYPE, const Type*, element_type) \
MkField(1, VALUES, Nodes, contents)

/// A value made out of more values.
/// Re-ordering values does not count as a computation here !
typedef struct Tuple_ Tuple;
#define Tuple_Fields(MkField) \
MkField(1, VALUES, Nodes, contents)

/// References either a global (yielding a pointer to it), or a constant (yielding a value of the type itself)
/// Declarations are not values themselves, this node is required to "convert" them.
typedef struct RefDecl_ RefDecl;
#define RefDecl_Fields(MkField) \
MkField(1, DECL, const Node*, decl)

/// Like RefDecl but for functions, it yields a _function pointer_ !
typedef struct FnAddr_ FnAddr;
#define FnAddr_Fields(MkField) \
MkField(1, DECL, const Node*, fn)

#define AnonLambda_Fields(MkField) \
MkField(1, VARIABLES, Nodes, params) \
MkField(1, TERMINATOR, const Node*, body)

#define BasicBlock_Fields(MkField) \
MkField(1, VARIABLES, Nodes, params) \
MkField(1, TERMINATOR, const Node*, body) \
MkField(1, DECL, const Node*, fn) \
MkField(1, STRING, String, name)

/// Populated by the parser for the bind pass, should be empty at all other times after that
/// (use the Scope analysis to figure out the real scope of a function)
typedef Nodes ChildrenBlocks;

typedef struct Function_ Function;
#define Function_Fields(MkField) \
MkField(1, VARIABLES, Nodes, params) \
MkField(1, TERMINATOR, const Node*, body) \
MkField(1, STRING, String, name) \
MkField(1, TYPES, Nodes, return_types) \
MkField(1, ANNOTATIONS, Nodes, annotations) \
MkField(1, BASIC_BLOCKS, ChildrenBlocks, children_blocks) \

#define Constant_Fields(MkField) \
MkField(1, ANNOTATIONS, Nodes, annotations) \
MkField(1, STRING, String, name) \
MkField(1, TYPE, const Type*, type_hint) \
MkField(0, VALUE, const Node*, value)

#define GlobalVariable_Fields(MkField) \
MkField(1, ANNOTATIONS, Nodes, annotations) \
MkField(1, TYPE, const Type*, type) \
MkField(1, STRING, String, name) \
MkField(1, POD, AddressSpace, address_space) \
MkField(0, VALUE, const Node*, init)

#define PrimOp_Fields(MkField) \
MkField(1, POD, Op, op) \
MkField(1, TYPES, Nodes, type_arguments) \
MkField(1, VALUES, Nodes, operands)

#define Call_Fields(MkField) \
MkField(1, VALUE, const Node*, callee) \
MkField(1, VALUES, Nodes, args)

// Those things are "meta" instructions, they contain other instructions.
// they map to SPIR-V structured control flow constructs directly
// they don't need merge blocks because they are instructions and so that is taken care of by the containing node

/// Structured "if" construct
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
MkField(1, TERMINATOR, const Node*, inside)

#define Let_Fields(MkField) \
MkField(1, INSTRUCTION, const Node*, instruction) \
MkField(1, ANON_LAMBDA, const Node*, tail)

#define LetMut_Fields(MkField) \
MkField(1, INSTRUCTION, const Node*, instruction) \
MkField(1, ANON_LAMBDA, const Node*, tail)

#define LetIndirect_Fields(MkField) \
MkField(1, INSTRUCTION, const Node*, instruction) \
MkField(1, VALUE, const Node*, tail)

typedef struct Jump_ Jump;
#define Jump_Fields(MkField) \
MkField(1, BASIC_BLOCK, const Node*, target) \
MkField(1, VALUES, Nodes, args)

/// A branch. Branches can cause divergence, but they can never cause re-convergence.
typedef struct Branch_ Branch;
#define Branch_Fields(MkField) \
MkField(1, VALUE, const Node*, branch_condition) \
MkField(1, BASIC_BLOCK, const Node*, true_target) \
MkField(1, BASIC_BLOCK, const Node*, false_target) \
MkField(1, VALUES, Nodes, args)

#define Switch_Fields(MkField) \
MkField(1, VALUE, const Node*, switch_value) \
MkField(1, VALUES, Nodes, case_values) \
MkField(1, BASIC_BLOCKS, Nodes, case_targets) \
MkField(1, BASIC_BLOCK, const Node*, default_target) \
MkField(1, VALUES, Nodes, args)

/// Join nodes are used to undo the divergence caused by branches. At join nodes, an explicit mask is used to force a number of divergent execution paths to resume.
/// If @p is_indirect is set, the target must be a function pointer. Otherwise, the target must be a function directly.
/// @p join_at _must_ be uniform.
typedef struct Join_ Join;
#define Join_Fields(MkField) \
MkField(1, VALUE, const Node*, join_point) \
MkField(1, VALUES, Nodes, args)

#define Return_Fields(MkField) \
MkField(1, DECL, const Node*, fn) \
MkField(1, VALUES, Nodes, args)

#define TailCall_Fields(MkField) \
MkField(1, VALUE, const Node*, target) \
MkField(1, VALUES, Nodes, args)

// typedef enum { Selection, Continue, Break } MergeConstructEnum;

/// These terminators are used in conjunction with structured constructs, they are used inside their bodies to yield a value
/// Using those terminators outside of an appropriate structured construct is undefined behaviour, and should probably be validated against
#define MergeSelection_Fields(MkField) \
MkField(1, VALUES, Nodes, args)

#define MergeContinue_Fields(MkField) \
MkField(1, VALUES, Nodes, args)

#define MergeBreak_Fields(MkField) \
MkField(1, VALUES, Nodes, args)

typedef enum {
    AnPayloadNone,
    AnPayloadValue,
    AnPayloadValues,
    AnPayloadMap,
} AnPayloadType;

typedef struct Annotation_ {
    const char* name;
    AnPayloadType payload_type;
    Strings labels;
    union {
        const Node* value;
        Nodes values;
    };
} Annotation;

#define WRITE_FIELD(hash, ft, t, n) t n;

#define CREATE_PAYLOAD_0(StructName, short_name)
#define CREATE_PAYLOAD_1(StructName, short_name) typedef struct StructName##_ { StructName##_Fields(WRITE_FIELD) } StructName;
#define CREATE_PAYLOAD(autogen_ctor, has_type_check_fn, has_payload, StructName, short_name) CREATE_PAYLOAD_##has_payload(StructName, short_name)
NODES(CREATE_PAYLOAD)

#undef WRITE_FIELD
#undef CREATE_PAYLOAD_0
#undef CREATE_PAYLOAD_1
#undef CREATE_PAYLOAD