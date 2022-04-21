## Shady language syntax

For the front-end mode, we can easily extend this by redefining `VALUE` as an arbitrary computation

```
PROGRAM := (DECL)*

DECL := const [TYPE] IDENTIFIER = VALUE // constant definition
        fn IDENTIFIER = RET_TYPE (PARAMS) BLOCK
        var ADDRESS_SPACE TYPE IDENTIFIER;

VAR := IDENTIFIER

PARAMS := ( [QTYPE IDENTIFIER [(, QTYPE IDENTIFIER)*]] )

BLOCK := { (INSTRUCTION;)* (CONTINUATION)* }

CONTINUATION := cont PARAMS BLOCK

INSTRUCTION := let IDENTIFIER [(, IDENTIFIER)*] = OP VALUES; // PrimOp
             | if VALUE then BLOCK else BLOCK;               // selection construct
             | while VALUE do BLOCK;                         // loop construct
             | return VALUES];

VALUE := VAR | LITERAL
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