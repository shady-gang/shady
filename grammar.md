## Shady language syntax

For the front-end mode, we can easily extend this by redefining `VALUE` as an arbitrary computation

```

VALUE := VAR_ID
       | LITERAL
       | tuple ( VALUES )
       | array (TYPE) ( [VALUES] )

VALUES := VALUE [, VALUES]

DECL := const [DATA_TYPE] IDENTIFIER = VALUE; // constant definition
        fn IDENTIFIER VALUE_TYPES PARAMS FN_BODY
        var ADDRESS_SPACE POINTABLE_TYPE IDENTIFIER;

PARAM := VALUE_TYPE VAR_ID
PARAMS := ( [PARAM [(, PARAM)+]] )

LOOP_PARAM := TYPE VAR = VALUE
LOOP_PARAMS := ( [LOOP_PARAM [(, LOOP_PARAM)+]] )

FN_BODY := { TERMINATOR; BASIC_BLOCK* } // the list of continuations is only for the front-end

BASIC_BLOCK := cont IDENTIFIER PARAMS BODY
LAMBDA := lambda PARAMS TERMINATOR

CALLEE := (DECL | VALUE) // calls can be direct or indirect

INSTRUCTION := PRIMOP `[` DATA_TYPES `]` (VALUES)                    // primop
             | call (VALUES) VALUES               // call
             | if TYPES (VALUE)               // structured if construct, can be defined to yield values
                   then LAMBDA
                   else LAMBDA
             | match TYPES (VALUE)           // structured match construct
                   (case LITERAL LAMBDA)* 
                   default LAMBDA
             | loop TYPES LOOP_PARAMS BODY  // structured loop construct
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
DATA_TYPE := int | float | ptr POINTABLE_TYPE | struct { (TYPE IDENTIFIER;)* }
DATA_TYPES: `[` DATA_TYPE [(, DATA_TYPE)+] `]`

VARIANCE_QUALIFIER = uniform | varying
VALUE_TYPE = VARIANCE_QUALIFIER D_TYPE
VALUE_TYPES = VALUE_TYPE [(, VALUE_TYPE)*]

POINTABLE_TYPE := DATA_TYPE
                | fn RET_TYPE ( [VALUE_TYPE [(, VALUE_TYPE)+]] )

OPAQUE_TYPE := POINTABLE_TYPE
             | texture
             | sampler

```

## Control flow instructions
