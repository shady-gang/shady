## Shady language syntax

For the front-end mode, we can easily extend this by redefining `VALUE` as an arbitrary computation

```

VAR := IDENTIFIER

VALUE := VAR
       | LITERAL
       | tuple ( VALUES )
       | array (TYPE) ( [VALUES] )

VALUES := VALUE [, VALUES]

DECL := const [TYPE] IDENTIFIER = VALUE; // constant definition
        fn IDENTIFIER Q_TYPES PARAMS FN_BODY
        var ADDRESS_SPACE TYPE IDENTIFIER;

PARAM := Q_TYPE VAR
PARAMS := ( [PARAM [(, PARAM)+]] )

PARAM := Q_TYPE VAR = VALUE
DEFAULT_PARAMS := ( [DEFAULT_PARAM [(, DEFAULT_PARAM)+]] )

FN_BODY := { TERMINATOR; BASIC_BLOCK* } // the list of continuations is only for the front-end

BASIC_BLOCK := cont IDENTIFIER PARAMS BODY
LAMBDA := lambda PARAMS TERMINATOR

CALLEE := (DECL | VALUE) // calls can be direct or indirect

TYPE_ARGS: `[` TYPE [(, TYPE)+] `]`

INSTRUCTION := PRIMOP [TYPE_ARGS] VALUES                    // primop
             | call (VALUES) VALUES               // call
             | if Q_TYPES (VALUE)               // structured if construct, can be defined to yield values
                   then LAMBDA
                   else LAMBDA
             | match Q_TYPES (VALUE)           // structured match construct
                   (case LITERAL LAMBDA)* 
                   default LAMBDA
             | loop Q_TYPES DEFAULT_PARAMS BODY  // structured loop construct
             | control LAMBDA                    // structured 'control' construct

TERMINATOR := unreachable;                           // use as a placeholder if nothing belongs. undefined behaviour if reached.
            | let INSTRUCTION in LAMBDA;             // alternative syntax: let PARAMS = INSTRUCTION; TERMINATOR
            | seq INSTRUCTION in VALUE;              // alternative syntax: let PARAMS = INSTRUCTION; TERMINATOR
            
            | tailcall VALUE VALUES;                // Start over in a new function, target is indirect (pointer), may be non-uniform
            | return VALUES;                          // return from current function
            
            | jump (IDENTIFIER) VALUES;               // one-way non-divergent branch, target is immediate and must be uniform
            | branch (OPERAND, ID, ID) VALUES;        // two-way divergent branch, targets are immediate and must be uniform
            | switch (OPERAND)                          // n-way divergent branch, targets are immediate and must be uniform
                (case LITERAL ID)* 
                default (ID);
            
            | join VALUE VALUES;                    // exits a control
            
            | merge VALUES;                           // Merges the current structured if/match construct
            | continue VALUES;                        // Structured continue
            | break VALUES;                           // Structured break

// things that have a non-opaque representation in memory
TYPE := int | float | ptr P_TYPE | struct { (TYPE IDENTIFIER;)* }

// such types can be pointed to by pointers
P_TYPE := DATA_TYPE | fn RET_TYPE ( [QTYPE [(, QTYPE)+]] )

// types with uniformity info
VARIANCE_Q = uniform | varying
QTYPE = VARIANCE_Q TYPE
Q_TYPES = Q_TYPE [(, Q_TYPE)*]

// optionally qualified types, the type inference can figure those out
MQTYPE = [VARIANCE_Q] TYPE
MQ_TYPES = MQ_TYPE [(, MQ_TYPE)*]

```

## Control flow instructions
