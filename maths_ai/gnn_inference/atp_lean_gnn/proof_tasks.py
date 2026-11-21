"""Original theorem boundaries shared by native search and independent replay.

Extracted from frozen GNN submission 1315dca; benchmark ingestion is separate.
"""

import re

NAME = r"(?:«[^»\r\n]+»|[^\W\d][\w'′]*)(?:\.(?:«[^»\r\n]+»|[^\W\d][\w'′]*))*"


def masked(source):
    chars, position, depth = list(source), 0, 0
    while position < len(source):
        if depth:
            if source.startswith("/-", position):
                depth += 1
                chars[position:position + 2] = "  "
                position += 2
            elif source.startswith("-/", position):
                depth -= 1
                chars[position:position + 2] = "  "
                position += 2
            else:
                if source[position] != "\n":
                    chars[position] = " "
                position += 1
        elif source.startswith("/-", position):
            depth = 1
            chars[position:position + 2] = "  "
            position += 2
        elif source.startswith("--", position):
            while position < len(source) and source[position] != "\n":
                chars[position] = " "
                position += 1
        elif source[position] == '"':
            chars[position] = " "
            position += 1
            while position < len(source):
                char = source[position]
                if char != "\n":
                    chars[position] = " "
                position += 1
                if char == "\\" and position < len(source):
                    chars[position] = " "
                    position += 1
                elif char == '"':
                    break
        else:
            position += 1
    if depth:
        raise ValueError("unterminated source comment")
    return "".join(chars)


def split_prefix(prefix):
    view, imports, spans = masked(prefix), [], []
    for match in re.finditer(r"^[ \t]*import[ \t]+([^\n]+)", view, re.MULTILINE):
        modules = match.group(1).strip().split()
        if not modules or any(not re.fullmatch(r"[A-Za-z_][\w.]*", module) for module in modules):
            raise ValueError("unsupported import declaration")
        imports.extend(modules)
        spans.append((match.start(), match.end()))
    chars = list(prefix)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    context = "".join(chars)
    if re.search(r"^\s*(?:namespace|section|set_option)\b[^\n]*\bin\s*$", masked(context), re.MULTILINE):
        raise ValueError("scoped command prefixes require a complete task context")
    return tuple(dict.fromkeys(imports)), context


def qualified_name(name, prefix):
    if name.startswith("_root_."):
        return name[7:]
    stack = []
    for line in masked(prefix).splitlines():
        match = re.fullmatch(rf"\s*(namespace|section|end)(?:\s+({NAME}))?\s*", line)
        if not match:
            continue
        kind, value = match.groups()
        if kind == "end":
            if not stack:
                raise ValueError("unbalanced source context")
            stack.pop()
        else:
            if kind == "namespace" and not value:
                raise ValueError("namespace name required")
            stack.append((kind, value))
    return ".".join([value for kind, value in stack if kind == "namespace"] + [name])


def checked_header(header, *, prefix=""):
    header = header.strip()
    if re.match(r"example\b", header):
        header = re.sub(r"^example\b", "theorem GnnSearchChecked", header, count=1)
    match = re.match(rf"(?:theorem|lemma)\s+({NAME})(?=\s|[:({{\[])", header)
    if not match or re.search(r"^[ \t]*(?:import|namespace|axiom|end)\b", masked(header), re.MULTILINE):
        raise ValueError("one named theorem/lemma header required")
    from maths_ai.lean_replay.lexer import tokenize, check_balanced
    tokens = tokenize(header)
    check_balanced(tokens)
    depth = 0
    for token in tokens:
        if token.kind == "open":
            depth += 1
        elif token.kind == "close":
            depth -= 1
        elif depth == 0 and token.text == ":=":
            raise ValueError("source header must not contain an existing proof body")
    return header, qualified_name(match.group(1), prefix)
