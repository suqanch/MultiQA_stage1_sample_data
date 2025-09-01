from pylatexenc.latex2text import (
    LatexNodes2Text, get_default_latex_context_db,
    EnvironmentTextSpec, MacroTextSpec
)
import re
import os

latex_paper_dir = "latex_paper_gpt3.5_turbo"

RULE_PREFIXES = ('\\toprule', '\\midrule', '\\bottomrule', '\\cmidrule')
def keep_tabular_header(node, l2tobj):
    body = l2tobj.nodelist_to_text(node.nodelist)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return ""
    candidates = []
    for ln in lines:
        if any(ln.startswith(pfx) for pfx in RULE_PREFIXES):
            continue
        if '&' in ln:
            candidates.append(ln)
    header = candidates[0] if candidates else lines[0]
    header = header.rstrip('\\').strip()
    return "[TableHeader] " + header

def caption_repl(node, l2tobj):
    """
    turn \caption{...} into [Caption] ...
    """
    if node.nodeargd is None or not node.nodeargd.argnlist:
        return "[Caption]"
    caption_text = l2tobj.nodelist_to_text(node.nodeargd.argnlist[0].nodelist)
    return "[Caption] " + caption_text.strip()


def label_repl(node, l2tobj):
    if node.nodeargd and node.nodeargd.argnlist:
        lab = l2tobj.nodelist_to_text(node.nodeargd.argnlist[0].nodelist).strip()
        return f'[Label id="{lab}"]'
    return "[Label]"

def ref_repl(node, l2tobj):
    if node.nodeargd and node.nodeargd.argnlist:
        tgt = l2tobj.nodelist_to_text(node.nodeargd.argnlist[0].nodelist).strip()
        return f'[Ref id="{tgt}"]'
    return "[Ref]"

def includegraphics_repl(node, l2tobj):
    '''file position'''
    src = ""
    if node.nodeargd and node.nodeargd.argnlist:
        argn = node.nodeargd.argnlist[-1]
        src = l2tobj.nodelist_to_text(argn.nodelist).strip()
    return f'[Graphic src="{src}"]' if src else "[Graphic]"

def table_env_repl(node, l2tobj):
    inner = l2tobj.nodelist_to_text(node.nodelist)
    return "[Table]" + inner

# def figure_env_repl(node, l2tobj):
#     inner = l2tobj.nodelist_to_text(node.nodelist)
#     return "[FigureBlock]\n" + inner + "\n[/FigureBlock]"

# def table_env_repl(node, l2tobj):
#     inner = l2tobj.nodelist_to_text(node.nodelist)
#     return "[TableBlock]\n" + inner + "\n[/TableBlock]"

def make_converter():
    ctx = get_default_latex_context_db()

    ctx.add_context_category('keep-label-ref', prepend=True, macros=[
        MacroTextSpec('label', simplify_repl=label_repl),
        MacroTextSpec('ref',   simplify_repl=ref_repl),
        MacroTextSpec('autoref', simplify_repl=ref_repl),
        MacroTextSpec('cref',  simplify_repl=ref_repl),
        MacroTextSpec('Cref',  simplify_repl=ref_repl),
    ])
    # redefine caption macro
    ctx.add_context_category(
        'keep-caption',
        prepend=True,
        macros=[
            MacroTextSpec('caption', simplify_repl=caption_repl),
            MacroTextSpec('captionof', simplify_repl=caption_repl),

        ]
    )
    ctx.add_context_category(
        'custom-graphics',
        prepend=True,
        macros=[
            MacroTextSpec('includegraphics', simplify_repl=includegraphics_repl),
            MacroTextSpec('includegraphics*', simplify_repl=includegraphics_repl),
        ]
    )


    # discard tabular content, keep only header
    ctx.add_context_category(
        'keep-tabular-header',
        prepend=True,
        environments=[
            EnvironmentTextSpec('tabular',    simplify_repl=keep_tabular_header),
            EnvironmentTextSpec('tabular*',   simplify_repl=keep_tabular_header),
            EnvironmentTextSpec('tabularx',   simplify_repl=keep_tabular_header),
            EnvironmentTextSpec('longtable',  simplify_repl=keep_tabular_header),
            EnvironmentTextSpec('array',      simplify_repl=keep_tabular_header),
        ]
    )

    ctx.add_context_category(
        'label-table-envs',
        prepend=True,
        environments=[
            EnvironmentTextSpec('table',   simplify_repl=table_env_repl),
            EnvironmentTextSpec('table*',  simplify_repl=table_env_repl),
            EnvironmentTextSpec('sidewaystable',  simplify_repl=table_env_repl),
            EnvironmentTextSpec('sidewaystable*', simplify_repl=table_env_repl),
        ]
    )

    # ctx.add_context_category('label-figure-envs', prepend=True, environments=[
    #     EnvironmentTextSpec('figure',   simplify_repl=figure_env_repl),
    #     EnvironmentTextSpec('figure*',  simplify_repl=figure_env_repl),
    # ])
    # ctx.add_context_category(
    #     'label-table-envs',
    #     prepend=True,
    #     environments=[
    #         EnvironmentTextSpec('table',   simplify_repl=table_env_repl),
    #         EnvironmentTextSpec('table*',  simplify_repl=table_env_repl),
    #         EnvironmentTextSpec('sidewaystable',  simplify_repl=table_env_repl),
    #         EnvironmentTextSpec('sidewaystable*', simplify_repl=table_env_repl),
    #     ]
    # )

    return LatexNodes2Text(
        latex_context=ctx,
        math_mode='with-delimiters',
        keep_comments=False,
        strict_latex_spaces='macros',
        keep_braced_groups=True,
    )

def merge_tex_files(file_path: str) -> str:
    """
    Recursively merge all .tex files under file_path.
    - Skip merged.tex to avoid merging the output back in
    - If main.tex exists at the top level, it will be placed first
    - The remaining files are sorted in relative path order
    """
    tex_files = []
    for root, _, files in os.walk(file_path):
        for f in files:
            if f.endswith(".tex") and f != "merged.tex":
                tex_files.append(os.path.join(root, f))

    if not tex_files:
        print(f"[skip] No .tex files in: {file_path}")
        return ""

    # Normal sorting
    tex_files.sort(key=lambda p: os.path.relpath(p, file_path))
    # print(tex_files)
    # If main.tex exists at the top level, it will be placed first
    top_main = os.path.join(file_path, "main.tex")
    if top_main in tex_files:
        tex_files.remove(top_main)
        tex_files.insert(0, top_main)

    parts = []
    for p in tex_files:
        rel = os.path.relpath(p, file_path)
        with open(p, "r", encoding="utf-8", errors="ignore") as infile:
            content = infile.read()
            content = "\n".join(re.sub(r'%.*$', '', line) for line in content.splitlines())
            parts.append(f"% ===== {rel} =====\n" + content)

    merged_text = "\n\n".join(parts) + "\n"
    output_file = os.path.join(file_path, "merged.tex")
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(merged_text)

    print(f"[merge] {len(tex_files)} files → {output_file}")
    return merged_text


# def latex_to_text(latex_str):
#     # Create a LatexNodes2Text object
#     latex2text = LatexNodes2Text(
#         math_mode='with-delimiters',
#         keep_comments=False,
#         strict_latex_spaces="macros",
#         keep_braced_groups=True
#     )
#     # Convert the LaTeX string to plain text
#     return latex2text.latex_to_text(latex_str)

def latex_to_text(latex_str):
    conv = make_converter()
    return conv.latex_to_text(latex_str)

DEF_PATTERN = re.compile(r'\\def\\([A-Za-z@]+)\{([^{}]*)\}')
def build_def_map(tex: str) -> dict:
    """
    from Latex extract \\def\\name{value}
    return { 'name': 'value' }
    """
    defs = {}
    for m in DEF_PATTERN.finditer(tex):
        name = m.group(1)
        value = m.group(2)
        value = value.replace(r'\xspace', '')
        defs[name] = value
    return defs

def apply_defs(tex: str, defs: dict) -> str:
    """
    \\name substitute to defs['name']
    """
    for name, val in sorted(defs.items(), key=lambda kv: -len(kv[0])):  
        tex = re.sub(rf'\\{re.escape(name)}(?![A-Za-z@])', val, tex)
    return tex

def main():

    # for each folder in latex_paper_dir
    for folder in os.listdir(latex_paper_dir):
        folder_path = os.path.join(latex_paper_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        latex_content = merge_tex_files(folder_path)
        defs = build_def_map(latex_content)
        out  = apply_defs(latex_content, defs)
        text_content = latex_to_text(out)
        output_file_path = os.path.join(folder_path, "output.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print("Conversion complete. Text written to:", output_file_path)

if __name__ == "__main__":
    main()