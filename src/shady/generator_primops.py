from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_primops_names_array(g, primops):
    g.g += "const char* primop_names[] = {\n"
    for node in primops:
        name = node["name"]
        g.g += f"\"{name}\","
    g.g += "\n};\n"

def generate(g, src):
    generate_header(g, src)

    primops = src["prim-ops"]
    generate_primops_names_array(g, primops)

    generate_bit_enum_classifier(g, "shd_get_primop_class", "OpClass", "Oc", "Op", "", "_op", primops)

main(generate)