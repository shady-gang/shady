from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_address_spaces(g, address_spaces):
    g.g += "typedef enum AddressSpace {\n"
    for address_space in address_spaces:
        name = address_space["name"]
        add_comments(g, "\t", address_space.get("description"))
        g.g += f"\tAs{name},\n"
    g.g += "\tNumAddressSpaces\n"
    g.g += "} AddressSpace;\n\n"

def generate_node_tags(g, nodes):
    g.g += "typedef enum {\n"
    g.g += "\tInvalidNode_TAG,\n"

    for node in nodes:
        name = node["name"]
        ops = node.get("ops")
        if ops is None:
            add_comments(g, "\t", node.get("description"))
        g.g += f"\t{name}_TAG,\n"
    g.g += "} NodeTag;\n\n"

def generate_primops(g, nodes):
    g.g += "typedef enum Op_ {\n"

    for node in nodes:
        name = node["name"]
        g.g += f"\t{name}_op,\n"

    g.g += "\tPRIMOPS_COUNT,\n"
    g.g += "} Op;\n"

def generate_node_tags_for_class(g, nodes, clazz, capitalized_class):
    g.g += "typedef enum {\n"
    if starts_with_vowel(clazz):
        g.g += f"\tNotAn{capitalized_class} = 0,\n"
    else:
        g.g += f"\tNotA{capitalized_class} = 0,\n"

    for node in nodes:
        name = node["name"]
        nclass = node.get("class")
        match(nclass):
            case None:
                pass
            case _ if type(nclass) is str:
                if nclass == clazz:
                    g.g += f"\t{capitalized_class}_{name}_TAG = {name}_TAG,\n"
                pass
            case _ if type(nclass) is list:
                for e in nclass:
                    if e == clazz:
                        g.g += f"\t{capitalized_class}_{name}_TAG = {name}_TAG,\n"
                pass
            case _:
                raise "Invalid datatype for a node's 'class' attribute"
    g.g += "}" + f" {capitalized_class}Tag;\n\n"

def generate(g, src):
    generate_header(g, src)

    generate_primops(g, src["prim-ops"])
    op_classes = src["prim-ops-classes"]
    generate_bit_enum(g, "OpClass", "Oc", op_classes, False)

    generate_address_spaces(g, src["address-spaces"])
    nodes = src["nodes"]
    generate_node_tags(g, nodes)

    #print(nodes)

    for node_class in src["node-classes"]:
        name = node_class["name"]
        generate_enum = node_class.get("generate-enum")
        capitalized = capitalize(name)

        if generate_enum is None or generate_enum == True:
            generate_node_tags_for_class(g, nodes, name, capitalized)

main(generate)