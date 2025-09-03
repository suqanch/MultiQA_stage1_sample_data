import os
import re
import json
from pre_tex_pipeline import latex_to_text, build_def_map, apply_defs
from pylatexenc.latex2text import (
    LatexNodes2Text, get_default_latex_context_db,
    EnvironmentTextSpec, MacroTextSpec
)

def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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


def parse_latex_sections(tex_path):
    """
    input: tex file path
    output: {section_title: text} dictionary and figure count
    """
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Build definition map
    def_map = build_def_map(content)

    # Apply definitions
    content = apply_defs(content, def_map)

    # \section{...}
    section_pattern = re.compile(r'\\section\*?\{([^}]*)\}', re.MULTILINE)
    sections = section_pattern.finditer(content)

    # {section: {
    #     content: ...,
    #     num_figures: ...
    # }
    # }

    section_dict = {}
    section_titles = []

    # Abstract, start index = 0
    section_titles.append(("Abstract", 0))

    # find all section titles and their positions
    for match in sections:
        section_titles.append((match.group(1), match.start()))

    # add document end marker
    section_titles.append(("__END__", len(content)))

    # split sections in order
    for i in range(len(section_titles) - 1):
        title = latex_to_text(section_titles[i][0]).strip()
        # reg a-zA-z0-9 and turn lower
        title = re.sub(r'[^a-zA-Z0-9 ]', '', title).lower()

        start = section_titles[i][1]
        end = section_titles[i+1][1]
        section_text_raw = content[start:end]
        section_text_clean = latex_to_text(section_text_raw).strip()
        # section_text_clean = re.sub(r'\s+', ' ', section_text_clean)

        
        # count table: [Table]
        # num_tables = len(re.findall(r'\s*\[Table\]\s*', section_text_clean))
        pattern_table = re.compile(r'\s*\[Table\]\s*', re.MULTILINE)
        num_tables = len(pattern_table.findall(section_text_clean))
        # print(re.findall(pattern_table, section_text_clean))
        # count figure: [Graphic src="..."]
        # num_figures = len(re.findall(r'Graphic', section_text_clean))
        pattern_graphic = re.compile(r'\[Graphic src="[^"]+"\]', re.MULTILINE)
        num_figures = len(pattern_graphic.findall(section_text_clean))


        # print(f"Section: {title}, Figures: {num_figures}, Tables: {num_tables}")
        section_dict[title] = {
            "content": section_text_clean,
            "multimodal_elements_num": num_figures + num_tables
        }
    # save_json(section_dict, save_json_path)
    return section_dict


if __name__ == "__main__":
    tex_file = "latex_paper_gpt4.1/MdocSum/merged.tex" 
    sections = parse_latex_sections(tex_file)

    # save json
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    print(f"Parsed sections saved to test.json")
