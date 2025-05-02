from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate(g, src):
    generate_header(g, src)

    nodes = src["nodes"]
    g.g += "void shd_visit_node_operands_generated(Visitor* visitor, NodeClass exclude, const Node* node) {\n"
    g.g += "\tswitch (node->tag) { \n"
    assert type(nodes) is list
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]

        g.g += f"\tcase {name}_TAG: {{\n"
        ops = node.get("ops")
        if ops is not None:
            assert type(ops) is list
            g.g += f"\t\tSHADY_UNUSED {name} payload = node->payload.{snake_name};\n"
            for op in ops:
                op_name = op["name"]
                is_list = op.get("list")
                ignore = op.get("ignore")
                if (ignore):
                    continue
                clazz = op.get("class")
                if clazz is None:
                    continue
                elif clazz == "string":
                    continue
                else:
                    class_cap = capitalize(clazz)
                    g.g += f"\t\tif ((exclude & Nc{class_cap}) == 0)\n"
                    if is_list:
                        g.g += f"\t\t\tshd_visit_ops(visitor, Nc{class_cap}, \"{op_name}\", payload.{op_name});\n"
                    else:
                        g.g += f"\t\t\tshd_visit_op(visitor, Nc{class_cap}, \"{op_name}\", payload.{op_name}, 0);\n"
        g.g += f"\t\tbreak;\n"
        g.g += f"\t}}\n"
    g.g += "\t\tdefault: assert(false);\n"
    g.g += "\t}\n"
    g.g += "}\n\n"

main(generate)