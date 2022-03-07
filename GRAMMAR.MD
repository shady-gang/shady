## Shady language syntax

For the front-end mode, we can easily extend this by redefining `VALUE` as an arbitrary computation

```
PROGRAM := (DECL)*

DECL := VAR_DECL | FN_DECL;

VAR_DECL := var MQTYPE IDENTIFIER ;
            var MQTYPE IDENTIFIER = VALUE;

FN_DECL := fn RET_TYPE IDENTIFIER PARAMS BLOCK

PARAMS := ( [QTYPE IDENTIFIER [(, QTYPE IDENTIFIER)*]] )

BLOCK := { (INSTRUCTION*) }

INSTRUCTION := let IDENTIFIER [(, IDENTIFIER)*] = COMP;
               VAR_DECL;
               IDENTIFIER := VALUE;
               if VALUE then BLOCK else BLOCK;
               while VALUE do BLOCK;
               return VALUE [(, VALUE)*];
               
COMP := OP VALUE+

VALUE := VAR | LITERAL

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