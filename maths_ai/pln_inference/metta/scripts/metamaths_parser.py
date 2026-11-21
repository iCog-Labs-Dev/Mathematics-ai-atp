import argparse

# 1. A lightweight S-Expression Parser to safely read nested parentheses
def parse_sexp(s):
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    tokens = s.split()
    
    def read_from_tokens(tokens):
        if len(tokens) == 0:
            raise SyntaxError('Unexpected EOF')
        token = tokens.pop(0)
        if token == '(':
            L = []
            while tokens[0] != ')':
                L.append(read_from_tokens(tokens))
            tokens.pop(0) # pop off ')'
            return L
        elif token == ')':
            raise SyntaxError('Unexpected )')
        else:
            return token
            
    return read_from_tokens(tokens)


def extract_top_level_forms(text):
    """Extract balanced top-level S-expressions from a text blob."""
    forms = []
    buf = []
    depth = 0
    started = False

    for ch in text:
        if ch == '(':
            depth += 1
            started = True
        if started:
            buf.append(ch)
        if ch == ')':
            depth -= 1
            if depth == 0 and started:
                forms.append("".join(buf))
                buf = []
                started = False
    return forms

# Helper to convert parsed python lists back to Lisp strings
def list_to_sexp(lst):
    if isinstance(lst, list):
        return "(" + " ".join(list_to_sexp(x) for x in lst) + ")"
    return str(lst)


INDENT = "   "


def formula_to_pettachainer(expr, level=0):
    """Render an object-language formula in PeTTaChainer's native rule format."""
    pad = INDENT * level

    if isinstance(expr, str):
        # The chainer examples wrap variables as atoms, e.g. ($phi).
        if expr.startswith('$'):
            return f"({expr})"
        return expr

    if not expr:
        return "()"

    head = expr[0]

    # Object-language implication from the porting file.
    if head == '→' and len(expr) == 3:
        premise = formula_to_pettachainer(expr[1], level + 2)
        conclusion = formula_to_pettachainer(expr[2], level + 2)
        return (
            f"(Implication \n"
            f"{INDENT * (level + 1)}(Premises {premise}) \n"
            f"{INDENT * (level + 1)}(Conclusions {conclusion})\n"
            f"{pad})"
        )

    # Negation in the chainer format used by test_same_cycle.metta.
    if head == '¬' and len(expr) == 2:
        return f"(Not {formula_to_pettachainer(expr[1], level)})"

    # Generic compound atom.
    return "(" + " ".join(formula_to_pettachainer(x, level) for x in expr) + ")"


def split_arrow_chain(expr):
    """Split a -> chain into (premises, conclusion). Supports n-ary and nested forms."""
    premises = []
    current = expr

    while isinstance(current, list) and current and current[0] == '->':
        if len(current) < 3:
            break

        # N-ary form: (-> A B C D) => premises A B C, conclusion D
        if len(current) > 3:
            premises.extend(current[1:-1])
            return premises, current[-1]

        # Binary nested form: (-> A (-> B C))
        premises.append(current[1])
        current = current[2]

    return premises, current


# 2. The Main Converter Logic
def convert_metamath_to_pettachainer(metamath_str):
    # Step A: Translate Greek variables to MeTTa variables
    replacements = {
        '𝜑': '$phi',     # phi
        '𝜓': '$psi',     # psi
        '𝜒': '$chi',     # chi
        '𝜃': '$theta',   # theta
        '𝜏': '$tau',     # tau
        '𝜂': '$eta',     # eta (likely the one from your ax-tr rule)
        '𝜆': '$lambda',  # lambda (matches the "cane with a flick" visually)
        '𝜁': '$zeta',    # zeta
        '𝜎': '$sigma',   # sigma
        '𝜇': '$mu',      # mu
        '𝛾': '$gamma',   # gamma
        '𝜌': '$rho'      # rho
    }
    for greek, ascii_var in replacements.items():
        metamath_str = metamath_str.replace(greek, ascii_var)
        
    try:
        parsed = parse_sexp(metamath_str)
    except Exception as e:
        return f";; Error parsing line: {e}"
    
    # Check if it matches the (MkIndexed <num> (...)) structure
    if not (isinstance(parsed, list) and parsed[0] == 'MkIndexed'):
        return ";; Unrecognized structure"
        
    inner = parsed[2]
    kind = inner[0]
    
    # Step B: Extract the name and the mathematical body
    if kind == 'MkAxiom':
        name = inner[1]
        body = inner[2]
    elif kind == 'MkTheorem':
        name = inner[1]
        body = inner[3] # Index 2 is the proof trace, which we skip
    else:
        return f";; Unrecognized kind: {kind}"
        
    # Step C: Process the Body (Meta-level vs Object-level)
    if isinstance(body, list) and body and body[0] == '->':
        # It's an inference rule with hypotheses.
        premises, conclusion = split_arrow_chain(body)
        
        # Format premises and conclusion
        premises_sexp = "\n            ".join(
            formula_to_pettachainer(p, 4) for p in premises
        )
        conclusion_sexp = formula_to_pettachainer(conclusion, 4)
        
        rule = f"""!(compileadd kb 
   (: (no_inverse {name}) 
      (Implication 
         (Premises 
            {premises_sexp}
         ) 
         (Conclusions 
            {conclusion_sexp}
         )
      ) 
      (STV 1.0 1.0)
   )
)"""
        return rule
    else:
        # It's a pure mathematical statement (no meta-level hypotheses)
        body_sexp = formula_to_pettachainer(body, 2)
        rule = f"""!(compileadd kb 
   (: (no_inverse {name}) 
      {body_sexp} 
      (STV 1.0 1.0)
   )
)"""
        return rule


def convert_file(input_path, output_path):
    """Batch convert all MkIndexed entries in a source file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        source = f.read()

    forms = extract_top_level_forms(source)
    converted = []

    for form in forms:
        try:
            parsed = parse_sexp(form)
        except Exception:
            continue

        if isinstance(parsed, list) and parsed and parsed[0] == 'MkIndexed':
            converted_rule = convert_metamath_to_pettachainer(form)
            if not converted_rule.startswith(';; Error') and not converted_rule.startswith(';; Unrecognized'):
                converted.append(converted_rule)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(';; Auto-generated by metamath_parsing.py — do not edit manually.\n')
        f.write(';; Generated from port_logic.metta using structural parsing.\n\n')
        f.write('\n'.join(converted))
        if converted:
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description='Convert Metamath MkIndexed forms to PeTTaChainer compileadd facts.'
    )
    parser.add_argument('--input', '-i', required=True, help='Path to input .metta file')
    parser.add_argument('--output', '-o', required=True, help='Path to output .metta file')
    args = parser.parse_args()

    convert_file(args.input, args.output)


if __name__ == '__main__':
    main()