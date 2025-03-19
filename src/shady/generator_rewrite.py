from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_can_be_default_rewritten_fn(g, nodes):
    g.g += "static bool can_be_default_rewritten(NodeTag tag) {\n"
    g.g += "\tswitch (tag) { \n"
    assert type(nodes) is list
    for node in nodes:
        if is_recursive_node(node):
            continue
        name = node["name"]
        g.g += f"\t\tcase {name}_TAG: return true;\n"
    g.g += "\t\tdefault: return false;\n"
    g.g += "\t}\n"
    g.g += "}\n\n"

def generate_rewriter_default_fns(g, nodes):
    g.g += "static const Node* recreate_node_identity_generated(Rewriter* rewriter, const Node* node) {\n"
    g.g += "\tswitch (node->tag) { \n"
    assert type(nodes) is list
    for node in nodes:
        if is_recursive_node(node):
            continue
        name = node["name"]
        snake_name = node["snake_name"]

        g.g += f"\t\tcase {name}_TAG: {{\n"
        ops = node.get("ops")
        if ops is not None:
            assert type(ops) is list
            g.g += f"\t\t\t{name} old_payload = node->payload.{snake_name};\n"
            g.g += f"\t\t\t{name} payload;\n"
            g.g += f"\t\t\tmemset(&payload, 0, sizeof(payload));\n"
            for op in ops:
                op_name = op["name"]
                is_list = op.get("list")
                ignore = op.get("ignore")
                if (ignore):
                    continue
                clazz = op.get("class")
                if clazz is None:
                    assert not is_list
                    g.g += f"\t\t\tpayload.{op_name} = old_payload.{op_name};\n"
                elif clazz == "string":
                    if is_list:
                        g.g += f"\t\t\tpayload.{op_name} = shd_strings(rewriter->dst_arena, old_payload.{op_name}.count, old_payload.{op_name}.strings);\n"
                    else:
                        g.g += f"\t\t\tpayload.{op_name} = shd_string(rewriter->dst_arena, old_payload.{op_name});\n"
                else:
                    class_cap = capitalize(clazz)
                    if is_list:
                        g.g += f"\t\t\tpayload.{op_name} = rewrite_ops_helper(rewriter, Nc{class_cap}, \"{op_name}\", old_payload.{op_name});\n"
                    else:
                        g.g += f"\t\t\tpayload.{op_name} = rewrite_op_helper(rewriter, Nc{class_cap}, \"{op_name}\", old_payload.{op_name});\n"
            g.g += f"\t\t\treturn {snake_name}(rewriter->dst_arena, payload);\n"
        else:
            g.g += f"\t\t\treturn {snake_name}(rewriter->dst_arena);\n"
        g.g += f"\t\t}}\n"
    g.g += "\t\tdefault: assert(false);\n"
    g.g += "\t}\n"
    g.g += "}\n\n"

def generate(g, src):
    generate_header(g, src)

    nodes = src["nodes"]
    generate_can_be_default_rewritten_fn(g, nodes)
    generate_rewriter_default_fns(g, nodes)

main(generate)