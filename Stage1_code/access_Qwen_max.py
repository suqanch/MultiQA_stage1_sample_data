import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import json5
# -------------------- Paths & Config --------------------

latex_paper_dir = Path("latex_paper_Qwen")
PROMPT_PATH = Path("myPromptV3/prompt1.txt")  # global prompt file

API_KEY = "sk"

# -------------------- Helpers --------------------

def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def strip_code_fences_to_json_str(s: str) -> str:
    """
    Remove ``` fences and language hints (e.g. ```json).
    Convert escaped newlines and then remove all newlines.
    """
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

# -------------------- OpenAI / DashScope --------------------

def ensure_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 北京区域；国际站改为 dashscope-intl
    )

# -------------------- Chat completion --------------------

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
        model="qwen-max-latest",   
        messages=messages,
        stream=False,
        # max_tokens=2048,
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

    # Traverse latex_paper/ and only process directories that contain output.txt
    for root, dirs, files in os.walk(latex_paper_dir):
        root_path = Path(root)
        print(root)
        if "output.txt" not in files:
            continue

        output_txt_path = root_path / "output.txt"
        doc_text = load_text(output_txt_path)

        result = run_prompt(api, prompt_text, doc_text, doc_name=str(output_txt_path.name))

        response_path = root_path / "prompt_response.json"
        save_json(response_path, result)
        print(f"[ok] saved: {response_path}")

        processed_path = root_path / "processed_response.json"
        process_response_text_to_json_file(result.get("response", ""), processed_path)
        print(f"[ok] saved: {processed_path}")

        # with open(root_path / "prompt_response.json", "r", encoding="utf-8") as f:
        #     result = json.load(f)
        # processed_path = root_path / "processed_response.json"
        # process_response_text_to_json_file(result.get("response", ""), processed_path)

    print("[done] all folders processed.")

if __name__ == "__main__":
    main()