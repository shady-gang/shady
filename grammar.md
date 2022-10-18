## Shady language syntax

For the front-end mode, we can easily extend this by redefining `VALUE` as an arbitrary computation

```
PROGRAM := (DECL)*

DECL := const [TYPE] IDENTIFIER = VALUE; // constant definition
        fn IDENTIFIER Q_TYPES (PARAMS) FN_BODY
        var ADDRESS_SPACE TYPE IDENTIFIER;

VAR := IDENTIFIER

VALUE := VAR
       | LITERAL
       | tuple ( VALUES )
       | array (TYPE) ( [VALUES] )

VALUES := VALUE [, VALUES]

OPERANDS := ( [QTYPE IDENTIFIER [(, QTYPE IDENTIFIER)*]] )

CONTINUATION := cont IDENTIFIER PARAMS BODY
LAMBDA := lambda PARAMS TERMINATOR

FN_BODY := { TERMINATOR; [CONTINUATION*] } // the list of continuations is only for the front-end

CALLEE := (DECL | VALUE) // calls can be direct or indirect

INSTRUCTION := PRIMOP(OPERANDS)                    // primop
             | call (OPERAND) OPERANDS            // call
             | if Q_TYPES (OPERAND)               // structured if construct, can be defined to yield values
                   then LAMBDA
                   else LAMBDA
             | match Q_TYPES (OPERAND)           // structured match construct
                   (case LITERAL LAMBDA)* 
                   default LAMBDA
             | loop Q_TYPES DEFAULT_PARAMS BODY  // structured loop construct
             | control LAMBDA                    // structured 'control' construct

TERMINATOR := unreachable;                              // use as a placeholder if nothing belongs. undefined behaviour if reached.
            | let INSTRUCTION in LAMBDA;                // alternative syntax: let PARAMS = INSTRUCTION; TERMINATOR
            
            | tailcall OPERAND OPERANDS;                // Start over in a new function, target is indirect (pointer), may be non-uniform
            | return OPERANDS;                          // return from current function
            
            | jump (IDENTIFIER) OPERANDS;               // one-way non-divergent branch, target is immediate and must be uniform
            | branch (OPERAND, ID, ID) OPERANDS;        // two-way divergent branch, targets are immediate and must be uniform
            | switch (OPERAND)                          // n-way divergent branch, targets are immediate and must be uniform
                (case LITERAL ID)* 
                default (ID);
            
            | join OPERAND OPERANDS;                    // exits a control
            
            | merge OPERANDS;                           // Merges the current structured if/match construct
            | continue OPERANDS;                        // Structured continue
            | break OPERANDS;                           // Structured break

// things that have a non-opaque representation in memory
TYPE := int | float | ptr P_TYPE | struct { (TYPE IDENTIFIER;)* }

// such types can be pointed to by pointers
P_TYPE := DATA_TYPE | fn RET_TYPE ( [QTYPE [(, QTYPE)*]] )

// types with uniformity info
VARIANCE_Q = uniform | varying
QTYPE = VARIANCE_Q TYPE
Q_TYPES = Q_TYPE [(, Q_TYPE)*]

// optionally qualified types, the type inference can figure those out
MQTYPE = [VARIANCE_Q] TYPE
MQ_TYPES = MQ_TYPE [(, MQ_TYPE)*]

```

## Control flow instructions
