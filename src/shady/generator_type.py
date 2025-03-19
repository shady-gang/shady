from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate(g, src):
    generate_header(g, src)

    nodes = src["nodes"]
    g.g += "const Type* _shd_check_type_generated(IrArena* a, const Node* node) {\n"
    g.g += "\tswitch(node->tag) {\n"
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]

        t = node.get("type")
        if t is None or t is True:
            g.g += f"\t\tcase {name}_TAG: "
            ops = node.get("ops")
            if ops is not None:
                g.g += f"return _shd_check_type_{snake_name}(a, node->payload.{snake_name});\n"
            else:
                g.g += f"return _shd_check_type_{snake_name}(a);\n"

    g.g += "\t\tdefault: return NULL;\n"
    g.g += "\t}\n"
    g.g += "}\n"

main(generate)