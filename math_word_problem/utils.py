# MIT License
# code by Soohwan Kim @sooftware

import re


def code_to_result(code: str):
    result = {}
    exec_format = "def func():\n"

    print_inner = re.findall(r'print(.+)', code)[0][1:-1]
    code = code.replace(f"print({print_inner})", f"return {print_inner}")

    lines = code.split('\n')

    for line in lines:
        exec_format += f"\t{line}\n"

    exec_format += "y = func()"
    exec(exec_format, globals(), result)
    return result["y"]
