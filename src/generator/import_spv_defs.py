import os.path
import sys
import json

from json_apply import *
from generator import *

def sanitize_node_name(name):
    is_type = False
    if name.startswith("OpType"):
        name = name[6:]
        is_type = True
    elif name.startswith("Op"):
        name = name[2:]

    if is_type:
        name += "Type"

    return name

def sanitize_field_name(name):
    if name[0] == '\'':
        name = name[1:-1]
    tmpname = ""
    for i in range(0, len(name)):
        if name[i] == ' ':
            tmpname += '_'
        else:
            tmpname += name[i].lower()
    return tmpname

def copy_object(dst, src, name, copied_name):
    o = src[name]
    dst[copied_name if copied_name is not None else name] = o

def apply_instruction_filter(filter, instruction, instantiated_filter, pending):
    match filter:
        case array if type(filter) is list:
            for f in array:
                apply_instruction_filter(f, instruction, instantiated_filter, pending)
        case object if type(filter) is dict:
            filter_name = filter.get("filter-name")
            if filter_name is not None:
                assert type(filter_name) is dict
                name = instruction["opname"]
                found = False
                for match_name, subfilter in filter_name.items():
                    if name == match_name:
                        found = True
                        pending.append(subfilter)
                if not found:
                    return

            json_apply_object(instantiated_filter, filter)
        case _:
            raise "Filters need to be arrays or objects"

def apply_instruction_filters(filter, instruction):
    instiated_filter = dict()
    pending = list()
    apply_instruction_filter(filter, instruction, instiated_filter, pending)
    #print("0:" + str(filter) + " f "+ str(instiated_filter))
    while len(pending) > 0:
        pending_filter = pending.pop(0)
        apply_instruction_filter(pending_filter, instruction, instiated_filter, pending)
        #print("1:" + str(instiated_filter))
    return instiated_filter

def apply_operand_filter(filter, operand, instantiated_filter, pending):
    #print(f"apply_operand_filter:{filter}")
    match filter:
        case array if type(filter) is list:
            for e in array:
                apply_operand_filter(e, operand, instantiated_filter, pending)
        case object if type(filter) is dict:
            filter_name = filter.get("filter-name")
            if filter_name is not None:
                assert type(filter_name) is dict
                name = operand.get("name")
                if name is None:
                    return
                found = False
                for match_name, subfilter in filter_name.items():
                    if name == match_name:
                        found = True
                        pending.append(subfilter)
                if not found:
                    return
            filter_kind = filter.get("filter-kind")
            if filter_kind is not None:
                assert type(filter_kind) is dict
                kind = operand["kind"]
                if kind is None:
                    kind = ""
                found = False
                for match_name, subfilter in filter_kind.items():
                    if match_name == kind:
                        found = True
                        pending.append(subfilter)
                if not found:
                    return

            json_apply_object(instantiated_filter, filter)

def apply_operand_filters(filter, instruction):
    instiated_filter = dict()
    pending = list()
    apply_operand_filter(filter, instruction, instiated_filter, pending)
    while len(pending) > 0:
        pending_filter = pending.pop(0)
        apply_operand_filter(pending_filter, instruction, instiated_filter, pending)
    return instiated_filter

def import_operand(operand, instruction_filter):
    kind = operand["kind"]
    assert kind is not None
    name = operand.get("name")
    if name is None:
        name = kind

    operand_filters = instruction_filter.get("operand-filters")
    assert operand_filters is not None
    filter = apply_operand_filters(operand_filters, operand)

    import_property = filter["import"]
    if import_property is None or import_property == "no":
        return
    elif import_property != "yes":
        print("a filter's 'import' property needs to be 'yes' or 'no'")

    field = dict()
    field_name = sanitize_field_name(name)
    field["name"] = field_name

    insert = filter.get("overlay")
    if insert is not None:
        json_apply_object(field, insert)

    return field

def import_filtered_instruction(instruction, filter):
    name = instruction["opname"]
    assert len(name) > 2

    import_property = filter.get("import")
    if import_property is None or import_property == "no":
        return
    elif import_property != "yes":
        print("a filter's 'import' property needs to be 'yes' or 'no'")

    node_name = sanitize_node_name(name)
    node = dict()
    node["name"] = node_name
    copy_object(node, instruction, "opcode", "spirv-opcode")

    insert = filter.get("overlay")
    if insert is not None:
        json_apply_object(node, insert)

    operands = instruction["operands"]
    ops = list()
    for operand in operands:
        field = import_operand(operand, filter)
        if field is not None:
            ops.append(field)

    if len(ops) > 0:
        node["ops"] = ops

    return node

def import_spirv_defs(imports, src, dst):
    spv = dict()
    dst["spv"] = spv
    copy_object(spv, src, "major_version", None)
    copy_object(spv, src, "minor_version", None)
    copy_object(spv, src, "revision", None)

    filters = imports["instruction-filters"]
    nodes = list()
    dst["nodes"] = nodes
    instructions = src["instructions"]
    for instruction in instructions:
        filter = apply_instruction_filters(filters, instruction)
        result = import_filtered_instruction(instruction, filter)
        if result is not None:
            assert(type(result) is dict)
            nodes.append(result)
    return

class JsonFile:
    contents = None
    root = None

ArgSelf = 0,
ArgDstFile = 1
ArgImportsFile = 2
ArgSpirvGrammarSearchPathBegins = 3

def shd_read_file(path):
    f = open(path)
    return f.read()

def shd_write_file(path, contents):
    f = open(path, "w")
    f.write(contents)

def main():
    argv = sys.argv[0:]
    argc = len(argv)

    assert argc > ArgSpirvGrammarSearchPathBegins
    dst_file = argv[ArgDstFile]
    spv_core_json_path = None
    for prefix in argv[ArgSpirvGrammarSearchPathBegins:]:
        path = f"{prefix}/spirv/unified1/spirv.core.grammar.json"
        print(f"trying path {path}")
        if os.path.exists(path):
            spv_core_json_path = path
            break

    if spv_core_json_path is None:
        raise "failed to find spirv.core.grammar.json"

    imports = JsonFile()
    imports.contents = shd_read_file(argv[ArgImportsFile])
    imports.root = json.loads(imports.contents)

    spirv = JsonFile()
    spirv.contents = shd_read_file(spv_core_json_path)
    spirv.root = json.loads(spirv.contents)

    print(f"Correctly opened json file: {spv_core_json_path}")

    output = dict()

    import_spirv_defs(imports.root, spirv.root, output)

    g = Growy()
    g.g += json.dumps(output)

    shd_write_file(dst_file, g.g)

main()