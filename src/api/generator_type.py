from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate(g, src):
    generate_header(g, src)

    nodes = src["nodes"]
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]
        assert snake_name is not None

        t = node.get("type")
        if t is None or t == True:
            ops = node.get("ops")
            if ops is not None:
                g.g += f"const Type* _shd_check_type_{snake_name}(IrArena*, {name});\n"
            else:
                g.g += f"const Type* _shd_check_type_{snake_name}(IrArena*);\n"

    g.g += "const Type* _shd_check_type_generated(IrArena* a, const Node* node);\n"

main(generate)