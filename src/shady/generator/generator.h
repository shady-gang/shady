#ifndef SHADY_GENERATOR_H
#define SHADY_GENERATOR_H

#include "growy.h"
#include "util.h"
#include "log.h"

#include <json-c/json.h>

#include <assert.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>

typedef const char* String;

void generate_header(Growy* g, json_object* root);
void add_comments(Growy* g, String indent, json_object* comments);
String to_snake_case(String camel);
String capitalize(String str);
bool starts_with_vowel(String str);

bool is_recursive_node(json_object* node);

json_object* lookup_node_class(json_object* src, String name);
bool find_in_set(json_object* node, String class_name);
String class_to_type(json_object* src, String class, bool list);
String get_type_for_operand(json_object* src, json_object* op);

void generate_bit_enum(Growy* g, String enum_type_name, String enum_case_prefix, json_object* cases);
void generate_bit_enum_classifier(Growy* g, String fn_name, String enum_type_name, String enum_case_prefix, String src_type_name, String src_case_prefix, String src_case_suffix, json_object* cases);

void json_apply_object(json_object* target, json_object* src);

#endif
