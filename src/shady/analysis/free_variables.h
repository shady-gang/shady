#ifndef SHADY_FREE_VARIABLES_H
#define SHADY_FREE_VARIABLES_H

#include "shady/ir.h"

typedef struct CFG_ CFG;
typedef struct CFNode_ CFNode;

typedef struct {
    CFNode* node;
    struct Dict* bound_by_dominators_set;
    struct Dict* bound_set;
    struct Dict* free_set;
    struct Dict* live_set;
} CFNodeVariables;

typedef enum {
    CfgVariablesAnalysisFlagNone = 0,
    CfgVariablesAnalysisFlagFreeSet = 0x1,
    CfgVariablesAnalysisFlagBoundSet = 0x2,
    CfgVariablesAnalysisFlagDomBoundSet = 0x4,
    CfgVariablesAnalysisFlagLiveSet = 0x8
} CfgVariablesAnalysisFlags;

struct Dict* compute_cfg_variables_map(const CFG* cfg, CfgVariablesAnalysisFlags flags);
void destroy_cfg_variables_map(struct Dict*);

#endif
