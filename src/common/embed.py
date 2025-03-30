import io
import sys

argv = sys.argv
argc = len(argv)

def usage_fail():
    print("Usage: embedder.py [string | bin] name src_file dst_file")
    exit(1)

if argc != 5:
    usage_fail()

global mode
if argv[1] == "string":
    mode = argv[1]
elif argv[1] == "bin":
    mode = argv[1]
else:
    usage_fail()

object_name = argv[2]

src = open(argv[3], "rb")
dst = open(argv[4], "wb")

src.seek(0, io.SEEK_END)
size = src.tell()
src.seek(0, 0)

print(f"source file is {size} bytes long")

buffer = bytearray(src.read())

dst.write(f"const char {object_name}[] = ".encode())
if mode == "string":
    dst.write(f"\"".encode())
    for pos in range(0, size):
        c = buffer[pos]
        if c == "\"".encode()[0]:
            dst.write("\\\"".encode())
        elif c == "\n".encode()[0]:
            dst.write("\\n\"\n\"".encode())
        else:
            dst.write(bytes([c]))
        #dst.write(bytes(c))
        #dst.write(f"{c}\\".encode())
    dst.write(f"\"".encode())
elif mode == "bin":
    dst.write("{".encode())
    for pos in range(0, size):
        c = int(buffer[pos])
        dst.write(f"{c}, ".encode())
    dst.write("}".encode())

dst.write(f";\n".encode())
dst.close()