#ifndef SHADY_FREE_FRONTIER_H
#define SHADY_FREE_FRONTIER_H

#include "shady/ir.h"
#include "shady/dict.h"
#include "cfg.h"
#include "scheduler.h"

NodeSet shd_free_frontier(Scheduler* scheduler, CFG* cfg, const Node* abs);

#endif
