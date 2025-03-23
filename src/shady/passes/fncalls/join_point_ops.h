#ifndef SHADY_JOIN_POINT_OPS_H
#define SHADY_JOIN_POINT_OPS_H

typedef enum {
    ShadyOpDefaultJoinPoint,
    ShadyOpCreateJoinPoint,
    ShadyOpDispatcherEnterFn,
    ShadyOpDispatcherContinue,
} ShadyJoinPointOpcodes;

#endif
