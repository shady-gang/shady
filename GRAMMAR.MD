## Shady language syntax

For the front-end mode, we can easily extend this by redefining `VALUE` as an arbitrary computation

```
PROGRAM := (DECL)*

DECL := def [TYPE] IDENTIFIER = VALUE // constant definition
        VAR_DECL

VAR_DECL := var ADDRESS_SPACE TYPE IDENTIFIER = VALUE;
            var ADDRESS_SPACE TYPE IDENTIFIER;

VAR := IDENTIFIER

FN_DECL := fn RET_TYPE PARAMS BLOCK

PARAMS := ( [QTYPE IDENTIFIER [(, QTYPE IDENTIFIER)*]] )

BLOCK := { (INSTRUCTION;)* }

INSTRUCTION := let IDENTIFIER [(, IDENTIFIER)*] = OP VALUES; // PrimOp
             | VAR_DECL;                                     // local variable
             | IDENTIFIER := VALUE;                          // stores
             | if VALUE then BLOCK else BLOCK;               // selection construct
             | while VALUE do BLOCK;                         // loop construct
             | return VALUES];

VALUE := VAR | LITERAL | FN_DECL
VALUES := VALUE+

TYPE := void | int | float | ptr DATA_TYPE | fn RET_TYPE ( [QTYPE [(, QTYPE)*]] )

DATA_TYPE := TYPE | struct { (TYPE IDENTIFIER;)* }

// functions may return multiple values
RET_TYPE = MQ_TYPE [(, MQ_TYPE)*]

VARIANCE_Q = uniform | varying

// qualified and maybe-qualified types
// maybe-qualified have inferrable unifornity
QTYPE = VARIANCE_Q TYPE
MQTYPE = [VARIANCE_Q] TYPE

```