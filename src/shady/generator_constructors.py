from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_pre_construction_validation(g, src):
    nodes = src["nodes"]
    g.g += "void pre_construction_validation(IrArena* arena, Node* node) {\n"
    g.g += "\tswitch (node->tag) { \n"
    assert type(nodes) is list
    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]

        g.g += f"\tcase {name}_TAG: {{\n"
        ops = node.get("ops")
        if ops is not None:
            assert type(ops) == list
            for op in ops:
                op_name = op["name"]
                clazz = op.get("class")
                if clazz is None:
                    continue
                is_list = op.get("list")
                if clazz == "string":
                    if is_list != True:
                        g.g += f"\t\tnode->payload.{snake_name}.{op_name} = shd_string(arena, node->payload.{snake_name}.{op_name});\n"
                    else:
                        g.g += f"\t\tnode->payload.{snake_name}.{op_name} = _shd_import_strings(arena, node->payload.{snake_name}.{op_name});\n"
                else:
                    classless = clazz == "none"
                    cap = capitalize(clazz)
                    g.g += "\t\t{\n"
                    extra = ""
                    if is_list == True:
                        g.g += f"\t\t\tsize_t ops_count = node->payload.{snake_name}.{op_name}.count;\n"
                        g.g += f"\t\t\tLARRAY(const Node*, ops, ops_count);\n"
                        g.g += f"\t\t\tif (ops_count > 0) memcpy(ops, node->payload.{snake_name}.{op_name}.nodes, sizeof(const Node*) * ops_count);\n"
                        g.g += f"\t\t\tfor (size_t i = 0; i < ops_count; i++) {{\n"
                        g.g += f"\t\t\tconst Node** pop = &ops[i];\n"
                        extra = "\t"
                    else:
                        g.g += f"\t\t\tconst Node** pop = &node->payload.{snake_name}.{op_name};\n"

                    if classless:
                        g.g += f"\t\t\t*pop = _shd_fold_node_operand({name}_TAG, 0, \"{op_name}\", *pop);\n"
                    else:
                        g.g += f"\t\t\t*pop = _shd_fold_node_operand({name}_TAG, Nc{cap}, \"{op_name}\", *pop);\n"

                    if not (op.get("nullable") or op.get("ignore")):
                        g.g += f"{extra}\t\t\tif (!*pop) {{\n"
                        g.g += f"{extra}\t\t\t\tshd_error(\"operand '{op_name}' of node '{name}' cannot be null\");\n"
                        g.g += f"{extra}\t\t\t}}\n"

                    if not classless:
                        g.g += f"{extra}\t\t\tif (arena->config.check_op_classes && *pop != NULL && !is_{clazz}(*pop)) {{\n"
                        g.g += f"{extra}\t\t\t\tshd_error_print(\"Invalid '{op_name}' operand for node '{name}', expected a {clazz}\");\n"
                        g.g += f"{extra}\t\t\t\tshd_error_die();\n"
                        g.g += f"{extra}\t\t\t}}\n"

                    if is_list == True:
                        g.g += "\t\t\t}\n"
                        g.g += f"\t\t\tnode->payload.{snake_name}.{op_name} = shd_nodes(arena, ops_count, ops);\n"

                    g.g += "\t\t}\n"
        g.g += "\t\tbreak;\n"
        g.g += "\t}\n"

    g.g += "\t\tdefault: break;\n"
    g.g += "\t}\n"
    g.g += "}\n\n"

def generate(g, src):
    generate_header(g, src)

    generate_pre_construction_validation(g, src)

main(generate)