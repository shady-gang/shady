from __future__ import absolute_import
from . generator import *

def is_recursive_node(node):
    return node.get("recursive") == True

def lookup_node_class(src, name):
    for c in src["node-classes"]:
        if c["name"] == name:
            return c
    return None

def class_to_type(src, clazz, list):
    assert clazz
    if clazz == "string":
        return "Strings" if list else "String"
    if clazz != "none" and lookup_node_class(src, clazz) is None:
        print(f"Invalid node class '{clazz}'")
    return "Nodes" if list else "const Node*"

def find_in_set(set_or_value, item):
    if set_or_value == item:
        return True
    if item in set_or_value:
        return True
    return False

def get_type_for_operand(src, op):
    op_type = op.get("type")
    list = op.get("list")
    if op_type is None:
        op_class = op["class"]
        assert op_class
        op_type = class_to_type(src, op_class, list)
    return op_type

def preprocess(src):
    for node in src["nodes"]:
        if node.get("snake_name") is None:
            node["snake_name"] = to_snake_case(node["name"])

def generate_bit_enum(g, enum_type_name, enum_case_prefix, cases, include_none):
    assert type(cases) is list
    g.g += "typedef enum {\n"
    if include_none:
        g.g += f"\t{enum_case_prefix}None = 0,\n"
    i = 1
    for node_class in cases:
        name = node_class["name"]
        capitalized = capitalize(name)
        g.g += f"\t{enum_case_prefix}{capitalized} = {hex(i)},\n"
        i *= 2
    g.g += "}" + f" {enum_type_name};\n\n"

def generate_bit_enum_classifier(g, fn_name, enum_type_name, enum_case_prefix, src_type_name, src_case_prefix, src_case_suffix, cases):
    g.g += f"{enum_type_name} {fn_name}({src_type_name} tag) {{\n"
    g.g += f"\tswitch (tag) {{ \n"
    assert type(cases) is list
    for node in cases:
        name = node["name"]
        g.g += f"\t\tcase {src_case_prefix}{name}{src_case_suffix}: \n"
        clazz = node.get("class")
        match(clazz):
            case None:
                g.g += "\t\t\treturn 0;\n"
                pass
            case _ if type(clazz) is str:
                cap = capitalize(clazz)
                g.g += f"\t\t\treturn {enum_case_prefix}{cap};\n"
                pass
            case _ if type(clazz) is list:
                g.g += "\t\t\treturn "
                for (j, clazzj) in zip(range(0, len(clazz)), clazz):
                    if j > 0:
                        g.g += " | "
                    cap = capitalize(clazzj)
                    g.g += f"{enum_case_prefix}{cap}"
                g.g += ";\n"
    g.g += f"\t\tdefault: assert(false);\n"
    g.g += f"\t}}\n"
    g.g += f"\tSHADY_UNREACHABLE;\n"
    g.g += f"}}\n"