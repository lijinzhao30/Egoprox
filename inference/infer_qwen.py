import os
import re
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from qwen_vl_utils import process_vision_info

def _to_file_uri(p: str, image_base_dir: str) -> str:
    if p.startswith(("file://", "http://", "https://")):
        return p
    if p.startswith("/"):
        return "file://" + p
    return "file://" + os.path.abspath(os.path.join(image_base_dir, p))

def ensure_file_scheme(messages: List[Dict[str, Any]], image_base_dir: str) -> List[Dict[str, Any]]:
    fixed = []
    for turn in messages:
        new_turn = {"role": turn.get("role"), "content": []}
        for item in turn.get("content", []):
            it = dict(item)
            typ = it.get("type")
            if typ in ("image", "video"):
                key = typ
                uri = it.get(key)
                if isinstance(uri, list):
                    it[key] = [_to_file_uri(x, image_base_dir) for x in uri]
                else:
                    it[key] = _to_file_uri(uri, image_base_dir)
            new_turn["content"].append(it)
        fixed.append(new_turn)
    return fixed

def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def chunk_iter(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

def inject_video_from_user_images(sample: Dict[str, Any], image_base_dir: str) -> List[Dict[str, Any]]:
    msgs = ensure_file_scheme(sample.get("data", []), image_base_dir)
    frames = []
    user_idx = None
    
    for ti, turn in enumerate(msgs):
        if turn.get("role") == "user" and isinstance(turn.get("content"), list):
            user_idx = ti
            for c in turn["content"]:
                if c.get("type") == "image" and isinstance(c.get("image"), str):
                    frames.append(c["image"])
            break

    if not frames or user_idx is None:
        return msgs

    video_elem = {"type": "video", "video": frames}
    texts_only = [c for c in msgs[user_idx]["content"] if c.get("type") == "text"]
    msgs[user_idx]["content"] = [video_elem] + texts_only
    return msgs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--image_base_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    samples = load_samples(args.input_json)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=False
    )

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    pbar = tqdm(total=len(samples), desc="Processing samples")
    
    for batch in chunk_iter(samples, args.batch_size):
        messages_batch = []
        is_chain_of_actions = []

        for sample in batch:
            is_chain_of_actions.append(sample.get("source", "") == "Chain of Actions")
            messages_batch.append(inject_video_from_user_images(sample, args.image_base_dir))

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages_batch, return_video_kwargs=True
        )
        
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10000,
                do_sample=False,
                temperature=None,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for sample, out_text, is_chain in zip(batch, output_texts, is_chain_of_actions):
            if is_chain:
                sample["answer"] = out_text.strip() if out_text else ""
            else:
                sample["answer"] = parse_answer_letter(out_text)
            results.append(sample)
            
        pbar.update(len(batch))

    pbar.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nInference finished. Processed {len(results)} samples.")
    print(f"Results successfully saved to: {os.path.abspath(args.output_json)}\n")

if __name__ == "__main__":
    main()