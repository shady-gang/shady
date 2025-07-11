#ifndef SHADY_IR_ANNOTATION_H
#define SHADY_IR_ANNOTATION_H

#include "shady/ir/base.h"

const Node* shd_lookup_annotation(const Node*, const char* name);
const Node* shd_get_annotation_value(const Node* annotation);
Nodes shd_get_annotation_values(const Node* annotation);
void shd_add_annotation(const Node*, const Node*);
void shd_add_annotation_named(const Node*, String);
void shd_remove_annotation_by_name(const Node*, String);
/// Gets the string literal attached to an annotation, if present.
const char* shd_get_annotation_string_payload(const Node* annotation);
bool shd_lookup_annotation_with_string_payload(const Node*, const char* annotation_name, const char* expected_payload);
Nodes shd_filter_out_annotation(IrArena* arena, Nodes annotations, const char* name);
Nodes shd_remove_annotation(IrArena* arena, Nodes annotations, int i);

#endif
