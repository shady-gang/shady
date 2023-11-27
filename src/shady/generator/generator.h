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
typedef struct {
    json_object* shd;
    json_object* spv;
} Data;

void generate(Growy* g, Data);

void generate_header(Growy* g, Data data);
void add_comments(Growy* g, String indent, json_object* comments);
String to_snake_case(String camel);
String capitalize(String str);
bool starts_with_vowel(String str);

bool has_custom_ctor(json_object* node);
void generate_node_ctor(Growy* g, json_object* nodes, bool definition);

void json_apply_object(json_object* target, json_object* src);

#endif
