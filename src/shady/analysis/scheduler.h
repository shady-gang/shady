#ifndef SHADY_SCHEDULER_H
#define SHADY_SCHEDULER_H

#include "shady/ir.h"
#include "cfg.h"

typedef struct Scheduler_ Scheduler;

Scheduler* new_scheduler(CFG*);
void destroy_scheduler(Scheduler*);

/// Returns the CFNode where that instruction should be placed, or NULL if it can be computed at the top-level
CFNode* schedule_instruction(Scheduler*, const Node*);

#endif
