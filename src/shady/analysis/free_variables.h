#ifndef SHADY_FREE_VARIABLES_H
#define SHADY_FREE_VARIABLES_H

#include "shady/ir.h"

typedef struct CFG_ CFG;
typedef struct CFNode_ CFNode;

typedef struct {
    CFNode* node;
    struct Dict* bound_set;
    struct Dict* free_set;
} CFNodeVariables;

struct Dict* compute_cfg_variables_map(const CFG* cfg);
void destroy_cfg_variables_map(struct Dict*);

#endif
