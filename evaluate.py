import json
import re
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict

import argparse

GROUP_KEYS = [
    "Intention_Approx",
    "Intention_Relative",
    "Exploration_Approx",
    "Exploration_Relative",
    "Exploitation_Approx",
    "Exploitation_Relative",
]

def pct(x: float) -> str:
    return f"{x * 100:.2f}%"

DIRECTION_LEGEND = {
    "A": "right",
    "B": "left",
    "C": "front",
    "D": "back",
    "E": "front-right",
    "F": "front-left",
    "G": "back-left",
    "H": "back-right",
}

RELAXED_NEIGHBORS = {
    "A": {"A", "E", "H"},
    "B": {"B", "F", "G"},
    "C": {"C", "E", "F"},
    "D": {"D", "G", "H"},
    "E": {"E", "A", "C"},
    "F": {"F", "B", "C"},
    "G": {"G", "B", "D"},
    "H": {"H", "A", "D"},
}

def _parse_node_token(x: Any) -> Optional[int]:
    if isinstance(x, bool): return None
    if isinstance(x, int): return int(x)
    if isinstance(x, float): return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s: return None
        m = re.search(r"(\d+)", s)
        if not m: return None
        return int(m.group(1))
    return None

def _extract_edges_from_any(obj: Any) -> List[str]:
    if isinstance(obj, list):
        out: List[str] = []
        for x in obj:
            if isinstance(x, str):
                out += re.findall(r"[A-H]", x.upper())
        return out
    if isinstance(obj, str):
        return re.findall(r"[A-H]", obj.upper())
    return []

def parse_llm_answer(
    llm_raw: Any,
    expected_node_len: int,
    expected_edge_len: int,
) -> Optional[Tuple[List[int], List[str]]]:
    if not isinstance(llm_raw, str):
        return None

    s = llm_raw.strip()
    if not s:
        return None

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list) and len(parsed) == 2 and isinstance(parsed[0], list):
            nodes = parsed[0]
            edges_obj = parsed[1]

            node_list: List[int] = []
            for x in nodes:
                v = _parse_node_token(x)
                if v is None:
                    raise ValueError(f"node token not parseable: {x!r}")
                node_list.append(v)

            if len(node_list) != expected_node_len:
                raise ValueError(f"node length mismatch: nodes={len(node_list)} (exp {expected_node_len})")

            edge_list = _extract_edges_from_any(edges_obj)
            if len(edge_list) > expected_edge_len:
                edge_list = edge_list[:expected_edge_len]

            return node_list, edge_list
        raise ValueError("parsed structure not [list, list]")
    except Exception:
        try:
            node_match = re.search(r"\[\s*\[\s*([^\]]+?)\s*\]\s*,", s)
            if not node_match:
                raise ValueError("node regex not match")

            node_str = node_match.group(1)
            node_nums = re.findall(r"\d+", node_str)
            node_list = [int(x) for x in node_nums]

            if len(node_list) != expected_node_len:
                raise ValueError(f"regex node length mismatch: nodes={len(node_list)} (exp {expected_node_len})")

            edge_list = re.findall(r"[A-H]", s.upper())
            if len(edge_list) > expected_edge_len:
                edge_list = edge_list[:expected_edge_len]

            return node_list, edge_list
        except Exception:
            return None

def find_matching_gt(answer_list: List[Dict[str, Any]], pred_nodes: List[int]) -> Optional[Dict[str, Any]]:
    for cand in answer_list:
        gt_nodes = cand.get("node", [])
        if isinstance(gt_nodes, list) and len(gt_nodes) == len(pred_nodes):
            try:
                if all(int(a) == int(b) for a, b in zip(gt_nodes, pred_nodes)):
                    return cand
            except Exception:
                continue
    return None

def strict_edge_accuracy(gt_edges: List[str], pred_edges: List[str]) -> float:
    if not gt_edges: return 1.0
    L = len(gt_edges)
    cmp_len = min(L, len(pred_edges))
    correct = 0
    for i in range(cmp_len):
        if str(pred_edges[i]).upper() == str(gt_edges[i]).upper():
            correct += 1
    return correct / float(L)

def relaxed_edge_accuracy(gt_edges: List[str], pred_edges: List[str]) -> float:
    if not gt_edges: return 1.0
    L = len(gt_edges)
    cmp_len = min(L, len(pred_edges))
    correct = 0
    for i in range(cmp_len):
        gt = str(gt_edges[i]).upper()
        pr = str(pred_edges[i]).upper()
        allowed = RELAXED_NEIGHBORS.get(gt, {gt})
        if pr in allowed:
            correct += 1
    return correct / float(L)

def eval_chain_of_actions(samples: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    total_eval = 0
    node_correct_count = 0
    node_correct_edge_acc_sum_strict = 0.0
    node_correct_edge_acc_sum_relaxed = 0.0
    node_correct_cnt_for_edge = 0

    for sample in samples:
        answer_list = sample.get("correct_answer", [])
        llm_raw = sample.get("answer", None)

        if not isinstance(answer_list, list) or len(answer_list) == 0:
            continue

        first_cand = answer_list[0]
        if not isinstance(first_cand, dict):
            continue

        gt_nodes0 = first_cand.get("node", [])
        gt_edges0 = first_cand.get("edge", [])

        if (
            not isinstance(gt_nodes0, list)
            or not isinstance(gt_edges0, list)
            or len(gt_nodes0) == 0
            or len(gt_edges0) == 0
        ):
            continue

        expected_node_len = len(gt_nodes0)
        expected_edge_len = len(gt_edges0)

        parsed = parse_llm_answer(llm_raw, expected_node_len, expected_edge_len)
        if parsed is None:
            continue

        pred_nodes, pred_edges = parsed
        total_eval += 1

        matched_gt = find_matching_gt(answer_list, pred_nodes)
        if matched_gt is None:
            continue

        node_correct_count += 1

        gt_edges = matched_gt.get("edge", [])
        if not isinstance(gt_edges, list):
            gt_edges = []

        gt_edges = [str(x).upper() for x in gt_edges]
        pred_edges = [str(x).upper() for x in pred_edges]

        node_correct_cnt_for_edge += 1

        edge_acc_strict = strict_edge_accuracy(gt_edges, pred_edges)
        node_correct_edge_acc_sum_strict += edge_acc_strict

        edge_acc_relaxed = relaxed_edge_accuracy(gt_edges, pred_edges)
        node_correct_edge_acc_sum_relaxed += edge_acc_relaxed

    node_acc_over_eval = (node_correct_count / float(total_eval)) if total_eval > 0 else 0.0

    if node_correct_cnt_for_edge > 0:
        edge_acc_given_node_strict = node_correct_edge_acc_sum_strict / float(node_correct_cnt_for_edge)
        edge_acc_given_node_relaxed = node_correct_edge_acc_sum_relaxed / float(node_correct_cnt_for_edge)
    else:
        edge_acc_given_node_strict = 0.0
        edge_acc_given_node_relaxed = 0.0

    return node_acc_over_eval, edge_acc_given_node_strict, edge_acc_given_node_relaxed

def main():
    parser = argparse.ArgumentParser(description="Evaluate Egoprox predictions")
    parser.add_argument("--path", type=str, required=True, help="Path to the JSON results file")
    args = parser.parse_args()
    
    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = OrderedDict((g, {"total": 0, "correct": 0}) for g in GROUP_KEYS)
    chain_of_actions_samples = []

    for item in data:
        src = item.get("source")
        
        if src == "Chain of Actions":
            chain_of_actions_samples.append(item)
            continue
        
        if src in stats:
            stats[src]["total"] += 1
            answer = item.get("answer")
            correct = item.get("correct_option")
            if answer == correct and answer is not None:
                stats[src]["correct"] += 1

    print("=========================================================================")
    print(f"{'Task':<24} Metrics")
    print("-------------------------------------------------------------------------")

    for grp, v in stats.items():
        t, c = v["total"], v["correct"]
        strict = "—" if t == 0 else pct(c / t)
        print(f"{grp:<24} ACC: {strict}")

    if chain_of_actions_samples:
        act_acc, rel_acc_s, rel_acc_l = eval_chain_of_actions(chain_of_actions_samples)
        chain_metrics = f"Act-Acc: {pct(act_acc)}, Rel-Acc-S: {pct(rel_acc_s)}, Rel-Acc-L: {pct(rel_acc_l)}"
        print(f"{'Chain of Actions':<24} {chain_metrics}")

    print("=========================================================================")

if __name__ == "__main__":
    main()