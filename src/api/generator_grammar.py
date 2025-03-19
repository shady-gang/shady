from src.generator.generator import *
from src.generator.generator_common import *
from src.generator.generator_main import main

def generate_node_payloads(g, src, nodes):
    for node in nodes:
        name = node["name"]

        ops = node.get("ops")
        if ops is not None:
            add_comments(g, "", node.get("description"))
            g.g += "typedef struct SHADY_DESIGNATED_INIT {\n"
            for op in ops:
                op_name = op["name"]
                g.g += f"\t{get_type_for_operand(src, op)} {op_name};\n"
            g.g += "}" + f" {name};\n\n"

def generate_node_type(g, nodes):
    g.g += "struct Node_ {\n"
    g.g += "\tIrArena* arena;\n"
    g.g += "\tNodeId id;\n"
    g.g += "\tconst Type* type;\n"
    g.g += "\tNodeTag tag;\n"
    g.g += "\tunion NodesUnion {\n"

    for node in nodes:
        name = node["name"]
        snake_name = node["snake_name"]

        ops = node.get("ops")
        if ops is not None:
            g.g += f"\t\t{name} {snake_name};\n"

    g.g += "\t} payload;\n"
    g.g += "\tNodes annotations;\n"
    g.g += "};\n\n"

def generate_isa_for_class(g, nodes, clazz, capitalized_class, use_enum):
    if use_enum:
        g.g += f"static inline {capitalized_class}Tag is_{clazz}(const Node* node) {{\n"
    else:
        g.g += f"static inline bool is_{clazz}(const Node* node) {{\n"
    g.g += f"\tif (shd_get_node_class_from_tag(node->tag) & Nc{capitalized_class})\n"
    if use_enum:
        g.g += f"\t\treturn ({capitalized_class}Tag) node->tag;\n"
        g.g += f"\treturn ({capitalized_class}Tag) 0;\n"
    else:
        g.g += f"\t\treturn true;\n"
        g.g += f"\treturn false;\n"
    g.g += "}\n\n"

def generate_header_getters_for_class(g, src, node_class):
    class_name = node_class["name"]
    class_ops = node_class.get("ops")
    if class_ops is None:
        return
    assert type(class_ops) is list
    for operand in class_ops:
        operand_name = operand["name"]
        g.g += f"{get_type_for_operand(src, operand)} get_{class_name}_{operand_name}(const Node* node);\n"

def generate_node_ctor(g, src, nodes):
    i = 0
    for node in nodes:
        name = node["name"]

        if is_recursive_node(node):
            continue

        if i > 0:
            g.g += "\n"
        i+=1

        snake_name = node["snake_name"]

        ops = node.get("ops")
        if ops is not None:
            g.g += f"static inline const Node* {snake_name}(IrArena* arena, {name} payload)"
        else:
            g.g += f"static inline const Node* {snake_name}(IrArena* arena)"

        g.g += " {\n"
        g.g += "\tNode node;\n"
        g.g += "\tmemset((void*) &node, 0, sizeof(Node));\n"
        g.g += "\tnode = (Node) {\n"
        g.g += "\t\t.arena = arena,\n"
        g.g += "\t\t.type = NULL,\n"
        g.g += f"\t\t.tag = {name}_TAG,\n"
        if ops is not None:
            g.g += f"\t\t.payload = {{ .{snake_name} = payload }},\n"
        g.g += "\t};\n"
        g.g += "\treturn _shd_create_node_helper(arena, node, NULL);\n"
        g.g += "}\n"

        # Generate helper variant
        if ops is not None:
            g.g += f"static inline const Node* {snake_name}_helper(IrArena* arena, "
            first = True
            for op in ops:
                op_name = op["name"]
                if op.get("ignore") == True:
                    continue
                if first:
                    first = False
                else:
                    g.g += ", "
                g.g += f"{get_type_for_operand(src, op)} {op_name}"
            g.g += ") {\n"
            g.g += f"\treturn {snake_name}(arena, ({name}) {{"
            first = True
            for op in ops:
                op_name = op["name"]
                if op.get("ignore") == True:
                    continue
                if first:
                    first = False
                else:
                    g.g += ", "
                g.g += f".{op_name} = {op_name}"
            g.g += "});\n"
            g.g += "}\n"

    g.g += "\n"

def generate_getters_for_class(g, src, nodes, node_class):
    class_name = node_class["name"]
    class_ops = node_class.get("ops")
    if class_ops is None:
        return
    assert type(class_ops) == list
    for operand in class_ops:
        operand_name = operand["name"]
        g.g += f"static inline {get_type_for_operand(src, operand)} get_{class_name}_{operand_name}(const Node* node) {{\n"
        g.g += "\tswitch(node->tag) {\n"
        for node in nodes:
            if find_in_set(node.get("class"), class_name):
                node_name = node["name"]
                g.g += f"\t\tcase {node_name}_TAG: "
                node_snake_name = node["snake_name"]
                g.g += f"return node->payload.{node_snake_name}.{operand_name};\n"
        g.g += "\t\tdefault: break;\n"
        g.g += "\t}\n"
        g.g += "\tassert(false);\n"
        g.g += "}\n\n"

def generate(g, src):
    generate_header(g, src)

    node_classes = src["node-classes"]
    generate_bit_enum(g, "NodeClass", "Nc", node_classes, True)

    nodes = src["nodes"]
    g.g += "NodeClass shd_get_node_class_from_tag(NodeTag tag);\n\n"
    generate_node_payloads(g, src, nodes)
    generate_node_type(g, nodes)

    g.g += "#include <string.h>\n"
    g.g += "#include <assert.h>\n"
    g.g += "Node* _shd_create_node_helper(IrArena* arena, Node node, bool* pfresh);\n"
    generate_node_ctor(g, src, nodes)

    for node_class in node_classes:
        name = node_class["name"]
        generate_enum = node_class.get("generate-enum")
        capitalized = capitalize(name)

        generate_getters_for_class(g, src, nodes, node_class)
        generate_isa_for_class(g, nodes, name, capitalized, generate_enum == None or generate_enum == True)

main(generate)