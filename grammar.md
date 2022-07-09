## Shady language syntax

For the front-end mode, we can easily extend this by redefining `VALUE` as an arbitrary computation

```
PROGRAM := (DECL)*

DECL := const [TYPE] IDENTIFIER = VALUE; // constant definition
        fn IDENTIFIER Q_TYPES (PARAMS) BLOCK
        var ADDRESS_SPACE TYPE IDENTIFIER;

VAR := IDENTIFIER

VALUE := VAR | LITERAL
VALUES := VALUE+

PARAMS := ( [QTYPE IDENTIFIER [(, QTYPE IDENTIFIER)*]] )

CONTINUATION := cont PARAMS BLOCK

BLOCK := { (LET;)* TERMINATOR; [CONTINUATION*] } // the list of continuations is only for the front-end

CALLEE := (DECL | VALUE) // calls can be direct or indirect

LET := let IDENTIFIER [(, IDENTIFIER)*] = INSTRUCTION; // Some instructions have results
     | INSTRUCTION;                                    // some don't

INSTRUCTION := PRIMOP VALUES
             | call CALLEE VALUES                        // direct-style call
             | if VALUE then BLOCK else (BLOCK | VALUES) // structured if statement
             | match VALUE                               // structured match statement
                   (case LITERAL BLOCK)* 
                   default (BLOCK | VALUES)
             | loop Q_TYPES PARAMS BLOCK VALUES         // structured loop statement

TERMINATOR := return VALUES                             // return from current function
            | unreachable                               // use as a placeholder if nothing belongs. undefined behaviour if reached.
            | jump CONTINUATION VALUES                  // unstructured jump instruction
            | br VALUE CONTINUATION CONTINUATION VALUES // unstructured branch instruction
            | callc CONTINUATION CALLEE VALUES          // call a function with a return continuation
            | callf DECL CALLEE VALUES                  // call a function with a return function
            | tailcall CALLEE VALUES                    // call a function 
            | join VALUES                               // yields to the innermost if/match statement
            | continue VALUES                           // jumps back the beginning of the current loop
            | break VALUES                              // jumps out of the current loop

TYPE := void | int | float | ptr DATA_TYPE | fn RET_TYPE ( [QTYPE [(, QTYPE)*]] )

DATA_TYPE := TYPE | struct { (TYPE IDENTIFIER;)* }

MQ_TYPES = MQ_TYPE [(, MQ_TYPE)*]
Q_TYPES = Q_TYPE [(, Q_TYPE)*]

VARIANCE_Q = uniform | varying

// qualified and maybe-qualified types
// maybe-qualified have inferrable unifornity
QTYPE = VARIANCE_Q TYPE
MQTYPE = [VARIANCE_Q] TYPE

```

## Control flow instructions

| Target   | Function call* | One-way jump         | One-way jump with sync |
|----------|----------------|----------------------|------------------------|
| direct   | callc          | br_jump / br_if_else | joinc                  |
| indirect | callf          | tailcall             | joinf                  |