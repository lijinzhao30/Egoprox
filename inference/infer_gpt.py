import os
import re
import json
import time
import tempfile
import base64
import uuid
import argparse
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import openai
from tqdm import tqdm

MAX_TOKENS  = 500
_thread_local = threading.local()

def get_thread_client(base_url: str, api_version: str, api_key: str):
    if not hasattr(_thread_local, "client"):
        _thread_local.client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            api_version=api_version,
            api_key=api_key,
        )
    return _thread_local.client

def gen_logid() -> str:
    return str(uuid.uuid4())

def call_llm(messages: List[Dict[str, Any]], base_url: str, api_version: str, api_key: str, model_name: str) -> str:
    client = get_thread_client(base_url, api_version, api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=MAX_TOKENS,
        extra_headers={"X-TT-LOGID": gen_logid()},
    )
    return resp.choices[0].message.content or ""

_ANS_RE = re.compile(
    r"(?:Therefore,\s*)?The\s+correct\s+answer\s+is\s*<?\s*([A-E])\s*>?\s*[\.\)]?",
    flags=re.IGNORECASE
)

def parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = list(_ANS_RE.finditer(text))
    if not m:
        return None
    return m[-1].group(1).upper()

def encode_image_to_base64(image_path: str, image_base_dir: str) -> str:
    from PIL import Image
    import io
    
    if image_path.startswith("file://"):
        image_path = image_path[7:]
    elif not image_path.startswith("/"):
        image_path = os.path.join(image_base_dir, image_path)
        
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {image_path}, {e}")

def build_messages_short_term(turns: List[Dict[str, Any]], image_base_dir: str) -> List[Dict[str, Any]]:
    mm = []
    for turn in turns:
        for item in turn.get("content", []):
            t = item.get("type")
            if t == "text":
                txt = item.get("text") or ""
                if txt:
                    mm.append({"type": "text", "text": txt})
            elif t == "image":
                path = item.get("image")
                if isinstance(path, str):
                    ext = "jpeg"
                    low = path.lower()
                    if low.endswith(".png"): ext = "png"
                    elif low.endswith(".webp"): ext = "webp"
                    b64 = encode_image_to_base64(path, image_base_dir)
                    mm.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64}"}
                    })
            elif t == "video":
                paths = item.get("video", [])
                if isinstance(paths, list):
                    for p in paths:
                        b64 = encode_image_to_base64(p, image_base_dir)
                        mm.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        })
    return [{"role": "user", "content": mm}]

def build_messages_long_term(sample: Dict[str, Any], image_base_dir: str) -> List[Dict[str, Any]]:
    system_prompt: str = sample.get("system_prompt", "") or ""
    user_prompt:   str = sample.get("user_prompt", "")   or ""
    image_paths:   List[str] = sample.get("image", []) or []

    user_content: List[Dict[str, Any]] = []
    for p in image_paths:
        b64 = encode_image_to_base64(p, image_base_dir)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
    if user_prompt.strip():
        user_content.append({"type": "text", "text": user_prompt})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": user_content},
    ]

def process_one(idx: int, sample: Dict[str, Any], image_base_dir: str, base_url: str, api_version: str, api_key: str, model_name: str) -> Tuple[int, Optional[str], Optional[str]]:
    try:
        is_chain = (sample.get("source", "") == "Chain of Actions")
        
        if is_chain:
            msgs = build_messages_long_term(sample, image_base_dir)
        else:
            turns = sample.get("data", [])
            msgs = build_messages_short_term(turns, image_base_dir)
            
        if not msgs or (not msgs[-1].get("content")):
            return idx, None, None

        text = call_llm(msgs, base_url, api_version, api_key, model_name)

        if is_chain:
            return idx, text.strip() if text else "", text
        else:
            letter = parse_answer_letter(text)
            return idx, letter, text
            
    except Exception as e:
        return idx, None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--image_base_dir", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--api_version", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    total = len(samples)
    
    final_list: List[Optional[Dict[str, Any]]] = [None] * total
    partial_by_id: Dict[Any, Dict[str, Any]] = {}
    
    if os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            partial_list = json.load(f)
        for obj in partial_list:
            sid = obj.get("id")
            if sid is not None:
                partial_by_id[sid] = obj

    to_run_indices: List[int] = []
    for idx, sample in enumerate(samples):
        sid = sample.get("id")
        if sid is not None and sid in partial_by_id and "answer" in partial_by_id[sid]:
            final_list[idx] = dict(partial_by_id[sid])
        else:
            to_run_indices.append(idx)

    if to_run_indices:
        pbar = tqdm(total=len(to_run_indices), desc="Processing samples")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {
                executor.submit(process_one, idx, samples[idx], args.image_base_dir, args.base_url, args.api_version, args.api_key, args.model_name): idx 
                for idx in to_run_indices
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    res_idx, answer, raw_text = fut.result()
                    obj = dict(samples[idx])
                    obj["answer"] = answer
                    final_list[idx] = obj
                except Exception:
                    obj = dict(samples[idx])
                    obj["answer"] = None
                    final_list[idx] = obj
                pbar.update(1)
        pbar.close()

    for idx, sample in enumerate(samples):
        if final_list[idx] is None:
            obj = dict(sample)
            obj["answer"] = None
            final_list[idx] = obj

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=os.path.dirname(args.output_json)) as tf:
        json.dump(final_list, tf, ensure_ascii=False, indent=2)
        tmp = tf.name
    os.replace(tmp, args.output_json)

    print(f"\nInference finished. Processed {total} samples.")
    print(f"Results successfully saved to: {os.path.abspath(args.output_json)}\n")

if __name__ == "__main__":
    main()