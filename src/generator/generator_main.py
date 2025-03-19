import json
import sys

from src.generator.generator import *
from src.generator.generator_common import preprocess
from src.generator.json_apply import json_apply_object

ArgSelf = 0,
ArgDstFile = 1
ArgFirstInput = 2

class JsonFile:
    contents = None
    root = None

def shd_read_file(path):
    f = open(path)
    return f.read()

def shd_write_file(path, contents):
    f = open(path, "w")
    f.write(contents)

def main(generate):
    argv = sys.argv
    argc = len(argv)

    dst_file = argv[ArgDstFile]

    json_files = []
    for path in argv[ArgFirstInput:]:
        json_file = JsonFile()
        json_file.contents = shd_read_file(path)
        json_file.root = json.loads(json_file.contents)
        json_files.append(json_file)
        print(f"Correctly opened json file: {path}")

    g = Growy()
    src = dict()

    for file in json_files:
        json_apply_object(src, file.root)

    preprocess(src)
    generate(g, src)

    shd_write_file(dst_file, g.g)