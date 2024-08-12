#ifndef SHADY_SLIM_H
#define SHADY_SLIM_H

typedef enum {
    SlimOpDereference,
    SlimOpAssign,
    SlimOpAddrOf,
    SlimOpSubscript,
    SlimOpBindVal,
    SlimOpBindVar,
    SlimOpBindContinuations,
    SlimOpUnbound,
} SlimFrontEndOpCodes;

#endif
