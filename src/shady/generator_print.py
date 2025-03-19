from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_node_print_fns(g, src):
    nodes = src["nodes"]
    g.g += "void _shd_print_node_generated(PrinterCtx* ctx, const Node* node) {\n"
    g.g += "\tswitch (node->tag) { \n"
    assert type(nodes) is list
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]

        g.g += f"\tcase {name}_TAG: {{\n"
        g.g += f"\t\tshd_print(ctx->printer, GREEN);\n"
        g.g += f"\t\tshd_print(ctx->printer, \"{name}\");\n"
        g.g += f"\t\tshd_print(ctx->printer, RESET);\n"
        g.g += f"\t\tshd_print(ctx->printer, \"(\");\n"
        ops = node.get("ops")
        if ops:
            assert type(ops) is list
            first = True
            for op in ops:
                ignore = op.get("ignore")
                if ignore:
                    continue
                tail = op.get("tail")

                if first:
                    first = False
                else:
                    g.g += "\t\tshd_print(ctx->printer, \", \");\n"

                op_name = op.get("name")
                op_class = op.get("class")
                if op_class is not None and op_class != "string":
                    is_list = op.get("list")
                    cap_class = capitalize(op_class)
                    if is_list:
                        g.g += "\t\t{\n"
                        g.g += f"\t\t\t_shd_print_node_operand_list(ctx, node, \"{op_name}\", Nc{cap_class}, node->payload.{snake_name}.{op_name});\n"
                        g.g += "\t\t}\n"
                    else:
                        g.g += "\t\t{\n"
                        g.g += f"\t\t\t_shd_print_node_operand(ctx, node, \"{op_name}\", Nc{cap_class}, node->payload.{snake_name}.{op_name});\n"
                        g.g += "\t\t}\n"
                else:
                    op_type = op.get("type")
                    if op_type is None:
                        assert op_class == "string"
                        is_list = op.get("list")
                        if is_list:
                            op_type = "Strings"
                        else:
                            op_type = "String"
                    s = ""
                    for c in op_type:
                        if c.isalpha() or c.isdigit():
                            s += c
                        else:
                            s += "_"
                    g.g += f"\t\t_shd_print_node_operand_{s}(ctx, node, \"{op_name}\", node->payload.{snake_name}.{op_name});\n"

        g.g += "\t\tshd_print(ctx->printer, \")\");\n"
        g.g += "\t\tbreak;\n"
        g.g += "\t}\n"

    g.g += "\t\tdefault: assert(false);\n"
    g.g += "\t}\n"
    g.g += "}\n"

def generate(g, src):
    generate_header(g, src)
    generate_node_print_fns(g, src)

main(generate)