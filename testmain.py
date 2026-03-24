import ast
with open("main.py") as f:
    ast.parse(f.read())
print("OK")
