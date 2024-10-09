#ifndef SHADY_TYPE_H
#define SHADY_TYPE_H

#include "shady/ir.h"

bool is_subtype(const Type* supertype, const Type* type);
void check_subtype(const Type* supertype, const Type* type);

bool is_arithm_type(const Type*);
bool is_shiftable_type(const Type*);
bool has_boolean_ops(const Type*);
bool is_comparable_type(const Type*);
bool is_ordered_type(const Type*);
bool is_physical_ptr_type(const Type* t);
bool is_generic_ptr_type(const Type* t);

bool is_reinterpret_cast_legal(const Type* src_type, const Type* dst_type);
bool is_conversion_legal(const Type* src_type, const Type* dst_type);

#include "type_generated.h"

#endif
