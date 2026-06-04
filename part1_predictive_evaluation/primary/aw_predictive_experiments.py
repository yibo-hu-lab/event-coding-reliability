
import os, sys, argparse, time, re
from pathlib import Path
import pandas as pd
import requests
from sklearn.metrics import f1_score

REPO       = str(Path(__file__).resolve().parents[1])
LLM_MODEL  = 'gemma2:9b'
OLLAMA_URL = 'http://localhost:11434/api/generate'

LABELS = ['COOPERATION', 'CONFLICT']
LABEL_MAP = {'COOPERATION': 0, 'CONFLICT': 1}

ALIASES = {
    'COOPERATIVE': 'COOPERATION', 'COOP': 'COOPERATION',
    'COOPERATE': 'COOPERATION', 'PEACE': 'COOPERATION',
    'CONFLICTUAL': 'CONFLICT', 'CONF': 'CONFLICT',
    'HOSTILE': 'CONFLICT', 'VIOLENCE': 'CONFLICT',
    'VERBAL_COOPERATION': 'COOPERATION', 'MATERIAL_COOPERATION': 'COOPERATION',
    'VERBAL_CONFLICT': 'CONFLICT', 'MATERIAL_CONFLICT': 'CONFLICT',
    'V_COOP': 'COOPERATION', 'M_COOP': 'COOPERATION',
    'V_CONF': 'CONFLICT', 'M_CONF': 'CONFLICT',
}

CODEBOOK = """1. COOPERATION: The source actor engages in cooperative behavior toward the target. This includes verbal cooperation (agreements, consultations, diplomatic support, promises) AND material cooperation (aid, trade, concessions, yielding, ceasefire, release of prisoners).

2. CONFLICT: The source actor engages in conflictual behavior toward the target. This includes verbal conflict (requests, demands, accusations, rejections, threats) AND material conflict (protests, sanctions, military mobilization, coercion, arrests, assaults, attacks, killings)."""

CODEBOOK_V2 = """COOPERATION: The source actor engages in cooperative behavior toward the target.

This includes:
- Verbal cooperation: agreements, consultations, diplomatic meetings, expressions of support, promises, signing or ratifying treaties
- Material cooperation: providing aid (monetary, military, humanitarian), trade, mutual exchanges, yielding, concessions, releasing prisoners, ceasefire, allowing access, easing restrictions

Examples of COOPERATION:
- <S>The US</S> provided $500M in humanitarian aid to <T>Yemen</T>. (material aid)
- <S>French officials</S> held talks with leaders of <T>Romania's</T> new government. (consultation)
- <S>The EU</S> signed a trade agreement with <T>Canada</T>. (diplomatic support)
- <S>A Taliban leader</S> surrendered in <T>Afghanistan</T>. (yielding)

CONFLICT: The source actor engages in conflictual behavior toward the target.

This includes:
- Verbal conflict: demands, accusations, criticisms, rejections, threats, warnings
- Material conflict: protests, demonstrations, sanctions, reducing relations, military mobilization, arrests, deportation, coercion, physical attacks, assaults, killings

Examples of CONFLICT:
- <S>Afghan rebels</S> kidnapped 16 <T>Soviet civilian advisers</T> and exploded bombs. (assault)
- <S>Thousands of workers</S> staged mass demonstrations against <T>the government</T>. (protest)
- <S>North Korea</S> issued a warning of serious consequences to <T>South Korea</T>. (threat)
- <S>A Brazilian federal court</S> rejected a request from <T>former President Lula</T>. (rejection)

DISAMBIGUATION RULES:
- If the action involves physical harm, killing, or attack, it is CONFLICT regardless of context.
- If the action involves arrests, deportation, sanctions, or coercion, it is CONFLICT.
- If the action involves signing agreements, providing aid, or holding meetings, it is COOPERATION.
- Future-tense cooperative actions (promises, agreements to do something) are COOPERATION.
- Ending or dissolving an organization is CONFLICT (loss/disruption).
- Founding or merging organizations is COOPERATION (constructive action).
- Personnel changes (resignations, firings, elections) should be classified based on whether the action is cooperative (appointment, taking office) or conflictual (firing, forced resignation)."""

ICL_EXAMPLES = """Example 1:
Sentence: <S>French National Assembly president Laurent Fabius</S> held talks with leaders of <T>Romania's</T> new government.
Answer: COOPERATION

Example 2:
Sentence: <S>Afghan rebels</S> have kidnapped 16 <T>Soviet civilian advisers</T> and exploded bombs in the capital.
Answer: CONFLICT

Example 3:
Sentence: <S>The US</S> provided $500M in humanitarian aid to <T>Yemen</T>.
Answer: COOPERATION

Example 4:
Sentence: <S>North Korea</S> issued a warning of serious consequences to <T>South Korea</T>.
Answer: CONFLICT

Example 5:
Sentence: <S>The EU</S> signed a trade agreement with <T>Canada</T>.
Answer: COOPERATION

Example 6:
Sentence: <S>Thousands of workers</S> staged mass demonstrations against <T>the government</T>.
Answer: CONFLICT

Example 7:
Sentence: <S>A Taliban leader</S> surrendered in <T>Afghanistan's Faryab province</T>.
Answer: COOPERATION

Example 8:
Sentence: <S>A Brazilian federal court</S> has rejected a request from <T>former President Lula da Silva</T>.
Answer: CONFLICT"""


def query_ollama(prompt, retries=3):
    for i in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                'model': LLM_MODEL, 'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.0, 'num_predict': 200}
            }, timeout=120)
            return r.json().get('response', '').strip()
        except Exception as e:
            if i < retries - 1:
                print(f"  Ollama retry {i+1}: {e}")
                time.sleep(3)
    return ''


def extract_binary(text):
    text_up = text.upper().strip()
    for label in LABELS:
        if text_up == label:
            return label
    for alias, canonical in ALIASES.items():
        if alias in text_up:
            return canonical
    for label in LABELS:
        if label in text_up:
            return label
    return 'UNKNOWN'


def extract_binary_cot(text):
    m = re.search(r'ANSWER:\s*(\w+)', text.upper())
    if m:
        candidate = m.group(1)
        for label in LABELS:
            if label == candidate:
                return label
        for alias, canonical in ALIASES.items():
            if alias == candidate:
                return canonical
    return extract_binary(text)


def compute_binary_f1(y_true_int, y_pred_int):
    return f1_score(y_true_int, y_pred_int, average='macro', zero_division=0) * 100


def load_aw_data(limit=None):
    path = f'{REPO}/datasets/AW_test.tsv'
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        sys.exit(1)
    df = pd.read_csv(path, sep='\t')
    valid_prefixes = ['Life','Conflict','Justice','Personnel','Contact','Transaction','Business']
    df = df[df['event_type'].apply(lambda et: any(et.startswith(p) for p in valid_prefixes))]
    df = df.reset_index(drop=True)
    df['source_clean'] = df['source'].apply(lambda x: str(x).split(":")[-1])
    df['target_clean'] = df['target'].apply(lambda x: str(x).split(":")[-1])
    if limit:
        df = df.head(limit)
    return df


def run_experiment(name, prompt_fn, save_path, extractor=extract_binary, limit=None):
    df = load_aw_data(limit)
    preds, truths, errors = [], [], 0

    print(f"\n  Running {name} on {len(df)} examples...")
    for idx, row in df.iterrows():
        sentence = str(row['marked_sentence'])
        gold = int(row['gold_binary'])
        response = query_ollama(prompt_fn(sentence))
        pred_label = extractor(response)
        if pred_label == 'UNKNOWN':
            errors += 1
            pred_label = 'CONFLICT'
        pred_int = LABEL_MAP[pred_label]
        preds.append(pred_int)
        truths.append(gold)
        if (idx + 1) % 100 == 0:
            f1 = compute_binary_f1(truths, preds)
            print(f"  [{idx+1}/{len(df)}] F1={f1:.1f} Unk={errors}")

    pd.DataFrame({
        'sentence': df['marked_sentence'].values[:len(preds)],
        'gold_binary': truths, 'pred_binary': preds,
        'event_type': df['event_type'].values[:len(preds)]
    }).to_csv(save_path, index=False)

    f1 = compute_binary_f1(truths, preds)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Binary F1={f1:.1f}")
    print(f"  Unknown={errors}/{len(df)}  Saved={save_path}")
    print(f"{'='*60}")
    return f1


def run_llm_no_codebook(limit=None):
    def prompt_fn(sentence):
        labels = ', '.join(LABELS)
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>).\n\n"
                f"Sentence: {sentence}\n\n"
                f"Choose one label from: {labels}\n\n"
                f"Output ONLY the label name, nothing else.")
    return run_experiment("LLM No Codebook", prompt_fn,
                          f'{REPO}/outputs/aw_llm_no_cb.csv', limit=limit)


def run_llm_compact(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier. Classify the relation "
                f"between source (<S></S>) and target (<T></T>).\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Output ONLY the label name (e.g. COOPERATION, CONFLICT), nothing else.")
    return run_experiment("LLM Compact", prompt_fn,
                          f'{REPO}/outputs/aw_llm_compact.csv', limit=limit)


def run_llm_cot_no_cb(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier.\n\n"
                f"Sentence: {sentence}\n\n"
                f"Think step by step:\n"
                f"1. Who is source, who is target?\n"
                f"2. What is the main action?\n"
                f"3. Verbal (statements/promises) or material (physical)?\n"
                f"4. Cooperative or conflictual?\n"
                f"5. Which label fits best: COOPERATION or CONFLICT?\n\n"
                f"After reasoning, write final answer as:\nANSWER: <label>")
    return run_experiment("LLM CoT (No CB)", prompt_fn,
                          f'{REPO}/outputs/aw_llm_cot_no_cb.csv',
                          extractor=extract_binary_cot, limit=limit)


def run_llm_icl(limit=None):
    def prompt_fn(sentence):
        labels = ', '.join(LABELS)
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>).\n\n"
                f"Here are labeled examples:\n\n{ICL_EXAMPLES}\n\n"
                f"Now classify:\nSentence: {sentence}\n\n"
                f"Labels: {labels}\nOutput ONLY the label name, nothing else.")
    return run_experiment("LLM ICL", prompt_fn,
                          f'{REPO}/outputs/aw_llm_icl.csv', limit=limit)


def run_llm_enriched(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier. Classify the relation "
                f"between source (<S></S>) and target (<T></T>).\n\n"
                f"{CODEBOOK_V2}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Output ONLY the label name (e.g. COOPERATION, CONFLICT), nothing else.")
    return run_experiment("LLM Enriched", prompt_fn,
                          f'{REPO}/outputs/aw_llm_enriched.csv', limit=limit)

def run_llm_cot_cb(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier.\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Think step by step:\n"
                f"1. Who is the source, who is the target?\n"
                f"2. What is the main action described?\n"
                f"3. Is this action verbal or material?\n"
                f"4. Is this cooperative or conflictual?\n"
                f"5. Based on the codebook definitions, which label fits best: "
                f"COOPERATION or CONFLICT?\n\n"
                f"ANSWER: ")
    return run_experiment("LLM CoT+CB", prompt_fn,
                          f'{REPO}/outputs/aw_llm_cot_cb.csv', limit=limit)


def print_table():
    print(f"\n{'='*60}")
    print(f"  AW RESULTS — AW_test (805 examples)")
    print(f"  Model: {LLM_MODEL}")
    print(f"{'='*60}")

    results = {}
    for name, csv_name in [
        ('LLM No Codebook', 'aw_llm_no_cb.csv'),
        ('LLM Compact', 'aw_llm_compact.csv'),
        ('LLM CoT (No CB)', 'aw_llm_cot_no_cb.csv'),
        ('LLM ICL', 'aw_llm_icl.csv'),
        ('LLM Enriched', 'aw_llm_enriched.csv'),
        ('LLM CoT+CB', 'aw_llm_cot_cb.csv'),
    ]:
        path = f'{REPO}/outputs/{csv_name}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            f1 = compute_binary_f1(df['gold_binary'].tolist(), df['pred_binary'].tolist())
            results[name] = f1
        else:
            results[name] = None

    print(f"\n{'Method':<30} {'Binary F1':>10}")
    print(f"{'─'*42}")
    for name, f1 in results.items():
        val = f"{f1:.1f}" if f1 is not None else "—"
        print(f"  {name:<28} {val:>8}")
    print(f"{'─'*42}")
    print(f"  {'[Prior] UP':<28} {'67.2':>8}")
    print(f"  {'[Prior] ZSP Tree':<28} {'88.0':>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True,
        choices=['llm_no_cb','llm_compact','llm_cot_no_cb','llm_icl','llm_enriched','llm_cot_cb',
                 'all','table'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    global LLM_MODEL
    if args.model:
        LLM_MODEL = args.model

    os.makedirs(f'{REPO}/outputs', exist_ok=True)

    dispatch = {
        'llm_no_cb':     lambda: run_llm_no_codebook(args.limit),
        'llm_compact':   lambda: run_llm_compact(args.limit),
        'llm_cot_no_cb': lambda: run_llm_cot_no_cb(args.limit),
        'llm_icl':       lambda: run_llm_icl(args.limit),
        'llm_enriched':  lambda: run_llm_enriched(args.limit),
        'llm_cot_cb':    lambda: run_llm_cot_cb(args.limit),
        'all':           lambda: [run_llm_no_codebook(args.limit),
                                  run_llm_compact(args.limit),
                                  run_llm_enriched(args.limit),
                                  run_llm_cot_no_cb(args.limit),
                                  run_llm_icl(args.limit),
                                  print_table()],
        'table':         lambda: print_table(),
    }
    dispatch[args.step]()

    if args.step not in ('table', 'all'):
        print(f"\n  → Run 'python3 primary/aw_predictive_experiments.py --step table' for comparison.")

if __name__ == '__main__':
    main()