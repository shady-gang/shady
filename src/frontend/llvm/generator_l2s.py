from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_llvm_shady_address_space_conversion(g, address_spaces):
    g.g += "AddressSpace l2s_convert_llvm_address_space(unsigned as) {\n"
    g.g += "\tstatic bool warned = false;\n"
    g.g += "\tswitch (as) {\n"
    for address_space in address_spaces:
        name = address_space["name"]
        llvm_id = address_space.get("llvm-id")
        if llvm_id is None:
            continue
        g.g += f"\t\t case {llvm_id}: return As{name};\n"
    g.g += "\t\tdefault:\n"
    g.g += "\t\t\tif (!warned)\n"
    g.g += "\t\t\t\tshd_warn_print(\"Warning: unrecognised address space %d\", as);\n"
    g.g += "\t\t\twarned = true;\n"
    g.g += "\t\t\treturn AsGeneric;\n"
    g.g += "\t}\n"
    g.g += "}\n"

def generate(g, src):
    generate_header(g, src)
    g.g += "#include \"l2s_private.h\"\n"
    g.g += "#include \"log.h\"\n"
    g.g += "#include <stdbool.h>\n"

    generate_llvm_shady_address_space_conversion(g, src["address-spaces"])

main(generate)