import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# -------------------- Paths & Config --------------------

latex_paper_dir = Path("latex_paper_gpt5")
PROMPT_PATH = Path("myPromptV3/prompt1.txt")

API_KEY = "sk-"

# -------------------- Helpers --------------------

def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def strip_code_fences_to_json_str(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s[3:].lstrip("json").lstrip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
    s = s.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "")
    return s

def process_response_text_to_json_file(resp_text: str, out_path: Path):
    cleaned = strip_code_fences_to_json_str(resp_text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        obj = {"raw": resp_text, "cleaned": cleaned, "_note": "not valid JSON"}
    save_json(out_path, obj)


def ensure_client() -> OpenAI:
    return OpenAI(api_key=API_KEY)

def build_user_content(prompt_text: str, doc_text: str, doc_name: str) -> str:
    return (
        f"{prompt_text}\n\n"
        f"---\n"
        f"[Document: {doc_name}]\n"
        f"{doc_text}\n"
        f"---\n"
        f"please answer the question based on the above document."
    )

def run_prompt(api: OpenAI, prompt_text: str, doc_text: str, doc_name: str) -> dict:
    user_content = build_user_content(prompt_text, doc_text, doc_name)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    completion = api.chat.completions.create(
        model="gpt-5",
        messages=messages,
        stream=False,
        store=True
    )
    content = completion.choices[0].message.content if completion.choices else ""
    usage = getattr(completion, "usage", None)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": completion.model,
        "response": content,
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage or {}),
    }

# -------------------- Main flow --------------------


def main():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

    api = ensure_client()
    prompt_text = load_text(PROMPT_PATH)

    # iterate latex_paper/*/output.txt only
    for output_txt_path in latex_paper_dir.glob("*/output.txt"):
        folder = output_txt_path.parent           # correct parent dir
        doc_name = folder.name                    # use folder name as doc_name

        try:
            print(f"processing {folder} ...")
            doc_text = load_text(output_txt_path)

            result = run_prompt(api, prompt_text, doc_text, doc_name=doc_name)

            response_path = folder / "prompt_response.json"
            save_json(response_path, result)
            print(f"[ok] saved: {response_path}")

            processed_path = folder / "processed_response.json"
            process_response_text_to_json_file(result.get("response", ""), processed_path)
            print(f"[ok] saved: {processed_path}")

        except Exception as e:
            # keep going even if one folder fails
            print(f"[error] {folder}: {e}")

    print("[done] all folders processed.")


if __name__ == "__main__":
    main()