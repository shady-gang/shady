#ifndef SHADY_FREE_FRONTIER_H
#define SHADY_FREE_FRONTIER_H

#include "shady/ir.h"
#include "cfg.h"
#include "scheduler.h"

struct Dict* free_frontier(Scheduler* scheduler, CFG*, const Node* abs);

#endif
