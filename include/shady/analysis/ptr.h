#ifndef SHADY_ANALYSIS_PTR_H
#define SHADY_ANALYSIS_PTR_H

#include "shady/ir/base.h"

#include "shady/analysis/uses.h"

typedef struct PtrAnalysis_ PtrAnalysis;

PtrAnalysis* shd_new_ptr_analysis(Module* module, const UsesMap* uses);
void shd_destroy_ptr_analysis(PtrAnalysis*);

typedef struct {
    const Type* type;
    /// Set when the alloca is used in a way the analysis cannot follow
    /// Allocation must be left alone in such cases!
    bool leaks;
    /// Set when the alloca is read from.
    bool read_from;
    /// Set when the alloca is used in a manner forbidden by logical pointer rules
    bool non_logical_use;

    //const Node* new;
} AllocaInfo;

const AllocaInfo* shd_analyze_alloc(PtrAnalysis*, const Node* old);

typedef struct {
    const AllocaInfo* src_alloca;
} PtrSourceKnowledge;

PtrSourceKnowledge shd_get_ptr_source_knowledge(PtrAnalysis*, const Node* ptr);

#endif
