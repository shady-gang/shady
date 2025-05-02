from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_node_names_strings_array(g, nodes):
    g.g += "const char* node_tags[] = {\n"
    g.g += "\t\"invalid\",\n"
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]
        g.g += f"\t\"{snake_name}\",\n"
    g.g += "};\n\n"

def generate_node_has_payload_array(g, nodes):
    g.g += "const bool node_type_has_payload[]  = {\n"
    g.g += "\tfalse,\n"
    for node in nodes:
        ops = node.get("ops")
        if ops is not None:
            g.g += "true,"
        else:
            g.g += "false,"
    g.g += "};\n\n"

def generate_node_is_recursive_array(g, nodes):
    g.g += "const bool node_type_is_recursive[]  = {\n"
    g.g += "\tfalse,\n"
    for node in nodes:
        if is_recursive_node(node):
            g.g += "true,"
        else:
            g.g += "false,"
    g.g += "};\n\n"

def generate_node_payload_hash_fn(g, src, nodes):
    g.g += "KeyHash _shd_hash_node_payload(const Node* node) {\n"
    g.g += "\tKeyHash hash = 0;\n"
    g.g += "\tswitch (node->tag) { \n"
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]

        ops = node.get("ops")
        if ops is not None:
            g.g += f"\tcase {name}_TAG: {{\n"
            g.g += f"\t\t{name} payload = node->payload.{snake_name};\n"
            for op in ops:
                op_name = op["name"]
                ignore = op.get("ignore")
                if not ignore:
                    g.g += f"\t\thash = hash ^ shd_hash(&payload.{op_name}, sizeof(payload.{op_name}));\n"
            g.g += "\t\tbreak;\n"
            g.g += "\t}\n"

    g.g += "\t\tdefault: assert(false);\n"
    g.g += "\t}\n"
    g.g += "\treturn hash;\n"
    g.g += "}\n"

def generate_node_payload_cmp_fn(g, src, nodes):
    g.g += "bool _shd_compare_node_payload(const Node* a, const Node* b) {\n"
    g.g += "\tbool eq = true;\n"
    g.g += "\tswitch (a->tag) { \n"
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]

        ops = node.get("ops")
        if ops is not None:
            g.g += f"\tcase {name}_TAG: {{\n"
            g.g += f"\t\t{name} payload_a = a->payload.{snake_name};\n"
            g.g += f"\t\t{name} payload_b = b->payload.{snake_name};\n"
            for op in ops:
                op_name = op["name"]
                ignore = op.get("ignore")
                if not ignore:
                    g.g += f"\t\teq &= memcmp(&payload_a.{op_name}, &payload_b.{op_name}, sizeof(payload_a.{op_name})) == 0;\n"
            g.g += "\t\tbreak;\n"
            g.g += "\t}\n"

    g.g += "\t\tdefault: assert(false);\n"
    g.g += "\t}\n"
    g.g += "\treturn eq;\n"
    g.g += "}\n"

def generate_node_is_nominal(g, nodes):
    g.g += "bool shd_is_node_nominal(const Node* node) {\n"
    g.g += "\tswitch (node->tag) { \n"
    for node in nodes:
        name = node["name"]
        if node.get("nominal"):
            g.g += f"\t\tcase {name}_TAG: return true;\n"
    g.g += "\t\tdefault: return false;\n"
    g.g += "\t}\n"
    g.g += "}\n"

def generate_address_space_name_fn(g, address_spaces):
    g.g += "String shd_get_address_space_name(AddressSpace as) {\n"
    g.g += "\tswitch (as) {\n"
    for address_space in address_spaces:
        name = address_space["name"]
        g.g += f"\t\tcase As{name}: return \"{name}\";\n"
    g.g += f"\t\tdefault : return NULL;\n"
    g.g += "\t}\n"
    g.g += "}\n"

def generate(g, src):
    generate_header(g, src)

    nodes = src["nodes"]
    generate_address_space_name_fn(g, src["address-spaces"])
    generate_node_names_strings_array(g, nodes)
    generate_node_is_nominal(g, nodes)
    generate_node_has_payload_array(g, nodes)
    generate_node_is_recursive_array(g, nodes)
    generate_node_payload_hash_fn(g, src, nodes)
    generate_node_payload_cmp_fn(g, src, nodes)
    generate_bit_enum_classifier(g, "shd_get_node_class_from_tag", "NodeClass", "Nc", "NodeTag", "", "_TAG", nodes)

    node_classes = src["node-classes"]
    for node_class in node_classes:
        name = node_class["name"]

        generate_enum = node_class.get("generate-enum")

main(generate)