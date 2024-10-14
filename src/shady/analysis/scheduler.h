#ifndef SHADY_SCHEDULER_H
#define SHADY_SCHEDULER_H

#include "shady/ir.h"
#include "cfg.h"

typedef struct Scheduler_ Scheduler;

Scheduler* shd_new_scheduler(CFG* cfg);
void shd_destroy_scheduler(Scheduler* s);

/// Returns the CFNode where that instruction should be placed, or NULL if it can be computed at the top-level
CFNode* shd_schedule_instruction(Scheduler* s, const Node* n);

#endif
