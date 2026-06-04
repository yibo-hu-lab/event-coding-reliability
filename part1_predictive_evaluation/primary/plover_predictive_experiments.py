#!/usr/bin/env python3
"""
plover_predictive_experiments.py — predictive PLV source-target event-coding runs.

Usage (run each step one at a time):
    python3 primary/plover_predictive_experiments.py --step tree
    python3 primary/plover_predictive_experiments.py --step tiny
    python3 primary/plover_predictive_experiments.py --step full
    python3 primary/plover_predictive_experiments.py --step llm_no_cb
    python3 primary/plover_predictive_experiments.py --step llm_compact
    python3 primary/plover_predictive_experiments.py --step llm_enriched
    python3 primary/plover_predictive_experiments.py --step llm_cot
    python3 primary/plover_predictive_experiments.py --step llm_icl
    python3 primary/plover_predictive_experiments.py --step table

Quick test (5 examples only):
    python3 primary/plover_predictive_experiments.py --step llm_enriched --limit 5
"""

import os, sys, argparse, subprocess, time, re
from pathlib import Path
import pandas as pd
import requests
from sklearn.metrics import f1_score

REPO       = str(Path(__file__).resolve().parents[1])
LLM_MODEL  = 'gemma2:9b'
OLLAMA_URL = 'http://localhost:11434/api/generate'

ROOTCODES = [
    'AGREE', 'CONSULT', 'SUPPORT', 'COOPERATE', 'AID', 'YIELD',
    'REQUEST', 'ACCUSE', 'REJECT', 'THREATEN',
    'PROTEST', 'SANCTION', 'MOBILIZE', 'COERCE', 'ASSAULT'
]
ROOT2QUAD = {
    'AGREE':1,'CONSULT':1,'SUPPORT':1,
    'COOPERATE':2,'AID':2,'YIELD':2,
    'REQUEST':3,'ACCUSE':3,'REJECT':3,'THREATEN':3,
    'PROTEST':4,'SANCTION':4,'MOBILIZE':4,'COERCE':4,'ASSAULT':4
}
ROOT2BIN = {r:(1 if ROOT2QUAD[r]<=2 else 2) for r in ROOTCODES}

CODEBOOK = """1. AGREE (Q1-Verbal Cooperation): Agree to, offer, promise, or indicate willingness to cooperate, including promises to sign or ratify agreements. Cooperative actions reported in future tense should be coded as AGREE.
2. CONSULT (Q1-Verbal Cooperation): All consultations and meetings, including visiting, hosting visits, meeting at neutral location, consultation by phone or other media.
3. SUPPORT (Q1-Verbal Cooperation): Initiate, resume, improve, or expand diplomatic, non-material cooperation; express support for, commend, approve, or ratify, sign, or finalize an agreement or treaty.
4. COOPERATE (Q2-Material Cooperation): Initiate, resume, improve, or expand mutual material cooperation or exchange, including economics, military, judicial matters, and sharing of intelligence.
5. AID (Q2-Material Cooperation): All provisions of material aid whose benefits primarily accrue to the recipient, including monetary, military, humanitarian, asylum etc.
6. YIELD (Q2-Material Cooperation): Yieldings or concessions: resignations, easing of restrictions, release of prisoners, repatriation, allowing access, disarming, ceasefire, military retreat.
7. REQUEST (Q3-Verbal Conflict): All verbal requests, demands, and orders, less forceful than threats. Demands as demonstrations/protests are coded as PROTEST.
8. ACCUSE (Q3-Verbal Conflict): Disapprovals, complaints; condemn, criticize, defame. Accuse, charge judicially or informally. Sue or bring to court. Investigations.
9. REJECT (Q3-Verbal Conflict): All rejections and refusals of assistance, policy changes, yielding, or meetings.
10. THREATEN (Q3-Verbal Conflict): All threats, coercive or forceful warnings with serious potential repercussions. Generally verbal acts.
11. PROTEST (Q4-Material Conflict): All civilian demonstrations and collective actions as protests. Dissent collectively, rally, gather to protest.
12. SANCTION (Q4-Material Conflict): All reductions in existing cooperative relations. Withdrawing or discontinuing diplomatic, commercial, or material exchanges.
13. MOBILIZE (Q4-Material Conflict): Military or police moves short of actual force. Demonstration of military capabilities. Distinct from verbal THREAT and actual ASSAULT.
14. COERCE (Q4-Material Conflict): Repression, restrictions on rights, coercive power short of violence: arresting, deporting, banning, curfew, cyber attacks.
15. ASSAULT (Q4-Material Conflict): Deliberate actions potentially resulting in substantial physical harm.

RULE: Prioritize Material Conflict (Q4) over Verbal Conflict (Q3). E.g. "protest to request" = PROTEST, "convict and arrest" = COERCE."""

CODEBOOK_V2 = """CLASSIFICATION GUIDE: Classify the political relation between source (<S></S>) and target (<T></T>) using the PLOVER ontology. Focus on the actual action described, not background context.

Q1 - VERBAL COOPERATION (statements, promises, diplomatic gestures):
1. AGREE: Express intent to cooperate, offer, promise, or indicate willingness. Includes promises to sign or ratify agreements. Future-tense cooperative actions = AGREE. Examples: "agreed to cooperate," "offered to negotiate," "promised aid," "will provide assistance."
2. CONSULT: All consultations and meetings. Visiting, hosting visits, meeting at neutral locations, phone or media consultations. Only use when the meeting itself is the primary action. Examples: "held talks," "met with officials," "consulted by phone."
3. SUPPORT: Initiate, resume, improve, or expand diplomatic or non-material cooperation. Express support for, commend, approve, ratify, sign, or finalize an agreement or treaty. Examples: "endorsed the plan," "signed a treaty," "praised the initiative."

Q2 - MATERIAL COOPERATION (physical actions, transfers, tangible concessions):
4. COOPERATE: Initiate, resume, improve, or expand mutual material cooperation or exchange. Includes economics, military cooperation, judicial matters, intelligence sharing. Examples: "joint military exercises," "trade agreement implemented," "shared intelligence."
5. AID: All provisions of material aid whose benefits primarily go to the recipient. Includes monetary, military, humanitarian, and asylum assistance. Examples: "provided $500M in aid," "sent humanitarian supplies," "granted asylum."
6. YIELD: Yieldings or concessions. Resignations, easing of restrictions, release of prisoners, repatriation, allowing access, disarming, ceasefire, military retreat. Examples: "released prisoners," "agreed to ceasefire," "withdrew troops," "lifted the ban," "resigned from office."

Q3 - VERBAL CONFLICT (statements, demands, accusations; NO physical action):
7. REQUEST: Verbal requests, demands, and orders, less forceful than threats. NOTE: Demands made as demonstrations/protests = PROTEST instead. Examples: "demanded an apology," "called for sanctions," "urged compliance."
8. ACCUSE: Disapprovals, complaints, condemnations, criticisms. Accuse or charge judicially or informally. Investigations. Examples: "condemned the attack," "charged with corruption," "launched an investigation."
9. REJECT: All rejections and refusals of assistance, policy changes, yielding, or meetings. Examples: "refused to negotiate," "rejected the proposal," "vetoed the resolution."
10. THREATEN: Threats, coercive or forceful warnings with serious repercussions. Verbal only. Examples: "warned of military action," "threatened sanctions," "issued an ultimatum."

Q4 - MATERIAL CONFLICT (physical actions, force, tangible punishment):
11. PROTEST: Civilian demonstrations and collective actions as protests. Rally, march, strike, gather to protest. Examples: "staged demonstrations," "marched in protest," "organized a strike."
12. SANCTION: Reductions in existing cooperative relations. Withdrawing or discontinuing diplomatic, commercial, or material exchanges. Examples: "imposed trade embargo," "recalled ambassador," "froze assets," "halted military aid."
13. MOBILIZE: Military or police moves short of actual force. Demonstration of military capabilities. Distinct from verbal THREAT and actual ASSAULT. Examples: "deployed troops to border," "placed forces on alert."
14. COERCE: Repression, restrictions on rights, coercive power short of violence. Arresting, deporting, banning, curfew, cyber attacks. Examples: "arrested opposition leaders," "imposed curfew," "banned the organization."
15. ASSAULT: Deliberate actions potentially resulting in substantial physical harm. Includes unconventional mass violence. Examples: "bombed the city," "kidnapped officials," "launched military offensive."

CRITICAL RULES:
- Material Conflict (Q4) overrides Verbal Conflict (Q3). "protest to request" = PROTEST. "convict and arrest" = COERCE.
- Future-tense cooperation = AGREE, not the material action. "agreed to provide aid" = AGREE. "will send troops to help" = AGREE.
- Negated or halted cooperation = SANCTION. "halted military aid" = SANCTION. "cut diplomatic ties" = SANCTION.
- Peacekeeping forces/workers/observers sent to help = AID, not MOBILIZE.
- CONSULT only when meeting is the main action, not a byproduct of another action."""

ICL_EXAMPLES = """Example 1:
Sentence: <S>French National Assembly president</S> held talks with leaders of <T>Romania's</T> new government.
Answer: CONSULT

Example 2:
Sentence: <S>The US</S> provided $500M in humanitarian aid to <T>Yemen</T>.
Answer: AID

Example 3:
Sentence: <S>A Brazilian federal court</S> rejected a request from <T>former President Lula</T>.
Answer: REJECT

Example 4:
Sentence: <S>Afghan rebels</S> kidnapped 16 <T>Soviet civilian advisers</T> and exploded bombs.
Answer: ASSAULT

Example 5:
Sentence: <S>A Taliban leader</S> surrendered in <T>Afghanistan's Faryab province</T>.
Answer: YIELD

Example 6:
Sentence: <S>North Korea</S> issued a warning of serious consequences to <T>South Korea</T>.
Answer: THREATEN

Example 7:
Sentence: <S>Thousands of workers</S> staged demonstrations against <T>the government</T>.
Answer: PROTEST

Example 8:
Sentence: <S>The EU</S> signed a trade agreement with <T>Canada</T>.
Answer: SUPPORT"""


# ================================================================
# HELPER FUNCTIONS
# ================================================================

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


def extract_label(text):
    text_up = text.upper().strip()
    for label in ROOTCODES:
        if text_up == label:
            return label
    m = re.match(r'^(\d{1,2})', text_up)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < 15:
            return ROOTCODES[idx]
    for label in ROOTCODES:
        if label in text_up:
            return label
    m = re.search(r'\b(1[0-5]|[1-9])\b', text)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < 15:
            return ROOTCODES[idx]
    return 'UNKNOWN'


def extract_label_cot(text):
    """Extract label from CoT response — look for ANSWER: line first."""
    m = re.search(r'ANSWER:\s*(\w+)', text.upper())
    if m:
        candidate = m.group(1)
        for label in ROOTCODES:
            if label == candidate:
                return label
    return extract_label(text)


def compute_f1(y_true, y_pred):
    root = f1_score(y_true, y_pred, average='macro',
                    labels=ROOTCODES, zero_division=0) * 100
    yq_t = [ROOT2QUAD.get(r, 0) for r in y_true]
    yq_p = [ROOT2QUAD.get(r, 0) for r in y_pred]
    quad = f1_score(yq_t, yq_p, average='macro', zero_division=0) * 100
    yb_t = [ROOT2BIN.get(r, 0) for r in y_true]
    yb_p = [ROOT2BIN.get(r, 0) for r in y_pred]
    binary = f1_score(yb_t, yb_p, average='macro', zero_division=0) * 100
    return binary, quad, root


def load_test_data(limit=None):
    df = pd.read_csv(f'{REPO}/datasets/PLV_test.tsv', sep='\t')
    if limit:
        df = df.head(limit)
    return df


def get_columns(df):
    sent_col = 'marked_sentence' if 'marked_sentence' in df.columns else df.columns[0]
    label_col = 'gold_root' if 'gold_root' in df.columns else df.columns[-1]
    return sent_col, label_col


def load_nli_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"  [MISSING] {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    y_true = df['gold_root'].astype(str).str.upper().tolist()
    best = None
    for suffix in ['L1', 'L2','L3']:
        col = f'root_{suffix}'
        if col in df.columns:
            y_pred = df[col].astype(str).str.upper().tolist()
            score = compute_f1(y_true, y_pred)
            print(f"  {suffix}: Binary={score[0]:.1f}  Quad={score[1]:.1f}  Root={score[2]:.1f}  Avg={sum(score)/3:.1f}")
            if best is None or sum(score)/3 > sum(best)/3:
                best = score
    return best


def run_llm_experiment(name, prompt_fn, save_path, extractor=extract_label, limit=None):
    df = load_test_data(limit)
    sent_col, label_col = get_columns(df)
    preds, truths, errors = [], [], 0

    print(f"\n  Running {name} on {len(df)} examples...")
    for idx, row in df.iterrows():
        sentence = str(row[sent_col])
        true_root = str(row[label_col]).upper().strip()
        response = query_ollama(prompt_fn(sentence))
        pred = extractor(response)
        if pred == 'UNKNOWN':
            errors += 1
            pred = 'REJECT'
        preds.append(pred)
        truths.append(true_root)
        if (idx + 1) % 100 == 0:
            b, q, r = compute_f1(truths, preds)
            print(f"  [{idx+1}/{len(df)}] Bin={b:.1f} Quad={q:.1f} Root={r:.1f} Unk={errors}")

    pd.DataFrame({
        'sentence': df[sent_col].values[:len(preds)],
        'gold_root': truths, 'pred_root': preds
    }).to_csv(save_path, index=False)

    b, q, r = compute_f1(truths, preds)
    avg = (b + q + r) / 3
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Binary={b:.1f}  Quad={q:.1f}  Root={r:.1f}  Avg={avg:.1f}")
    print(f"  Unknown={errors}/{len(df)}  Saved={save_path}")
    print(f"{'='*60}")
    return b, q, r


# ================================================================
# STEP RUNNERS
# ================================================================

def run_zsp(name, prompt, score_name, output_name, online=False):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    npy = f'{REPO}/scores/{score_name}'
    out = f'{REPO}/outputs/{output_name}'

    if online:
        setting, run_nli = 'online', ''
    else:
        if not os.path.exists(npy):
            print(f"  ERROR: {npy} not found! Run with online mode instead.")
            return None
        setting, run_nli = 'offline', '--run_offline_nli False'

    cmd = (
        f"cd {REPO} && python3 legacy_nli/main_script.py "
        f"--data_dir ./datasets/PLV_test.tsv "
        f"--prompt_dir ./prompts/{prompt} "
        f"--score_dir ./scores/{score_name} "
        f"--model_name roberta-large-mnli "
        f"--output_dir ./outputs/{output_name} "
        f"--consult_penalty 0.02 "
        f"--infer_setting {setting} "
        f"{run_nli} "
        f"--infer_details True "
        f"--summary_details True"
    )
    print(f"  CMD: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  WARNING: main_script.py exited with code {ret}")
    return load_nli_results(out)


def run_zsp_tree():
    return run_zsp("ZSP Tree (online, GPU)", "Tree.txt",
                   "PLV_test-Tree.npy", "PLV_test-Tree-result.csv", online=True)

def run_zsp_tiny():
    return run_zsp("ZSP Tiny (online, GPU)", "Tiny.txt",
                   "PLV_test-Tiny.npy", "PLV_test-Tiny-result.csv", online=True)

def run_zsp_full():
    return run_zsp("ZSP Full (online, GPU)", "Full.txt",
                   "PLV_test-Full.npy", "PLV_test-Full-result.csv", online=True)


def run_llm_no_codebook(limit=None):
    def prompt_fn(sentence):
        labels = ', '.join(ROOTCODES)
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>).\n\n"
                f"Sentence: {sentence}\n\n"
                f"Choose one label from: {labels}\n\n"
                f"Output ONLY the label name, nothing else.")
    return run_llm_experiment("LLM No Codebook", prompt_fn,
                              f'{REPO}/outputs/llm_no_codebook.csv', limit=limit)

def run_llm_compact(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier. Classify the relation "
                f"between source (<S></S>) and target (<T></T>).\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Output ONLY the label name (e.g. AGREE, ASSAULT), nothing else.")
    return run_llm_experiment("LLM Compact", prompt_fn,
                              f'{REPO}/outputs/llm_compact.csv', limit=limit)

def run_llm_cot(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier.\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Think step by step:\n"
                f"1. Who is source, who is target?\n"
                f"2. What is the main action?\n"
                f"3. Verbal (statements/promises) or material (physical)?\n"
                f"4. Cooperative or conflictual?\n"
                f"5. Which label fits best?\n\n"
                f"After reasoning, write final answer as:\nANSWER: <label>")
    return run_llm_experiment("LLM CoT", prompt_fn,
                              f'{REPO}/outputs/llm_cot.csv',
                              extractor=extract_label_cot, limit=limit)

def run_llm_icl(limit=None):
    def prompt_fn(sentence):
        labels = ', '.join(ROOTCODES)
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>).\n\n"
                f"Here are labeled examples:\n\n{ICL_EXAMPLES}\n\n"
                f"Now classify:\nSentence: {sentence}\n\n"
                f"Labels: {labels}\nOutput ONLY the label name, nothing else.")
    return run_llm_experiment("LLM ICL", prompt_fn,
                              f'{REPO}/outputs/llm_icl.csv', limit=limit)

def run_llm_enriched(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier. Classify the relation "
                f"between source (<S></S>) and target (<T></T>).\n\n"
                f"{CODEBOOK_V2}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Output ONLY the label name (e.g. AGREE, ASSAULT), nothing else.")
    return run_llm_experiment("LLM Enriched", prompt_fn,
                              f'{REPO}/outputs/llm_enriched.csv', limit=limit)
import json
def load_json_codebook():
    with open(f'{REPO}/plover_codebook.json', 'r') as f:
        return json.load(f)

def format_json_codebook_for_prompt(cb):
    lines = []
    lines.append(cb['task_description'])
    lines.append("")
    for entry in cb['labels']:
        lines.append(f"Label: {entry['label']}")
        lines.append(f"  Quadcode: {entry['quadcode']}")
        lines.append(f"  Definition: {entry['definition']}")
        lines.append(f"  Clarification: {entry['clarification']}")
        lines.append("")
    lines.append("RULES:")
    for rule in cb['disambiguation_rules']:
        lines.append(f"  - {rule}")
    lines.append("")
    lines.append(cb['output_reminder'])
    return '\n'.join(lines)

def run_llm_with_json_codebook(limit=None):
    cb = load_json_codebook()
    codebook_prompt = format_json_codebook_for_prompt(cb)
    def prompt_fn(sentence):
        return (f"{codebook_prompt}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Label:")
    return run_llm_experiment("LLM JSON CB", prompt_fn,
                              f'{REPO}/outputs/llm_json_cb.csv', limit=limit)


# ================================================================
# FINAL TABLE
# ================================================================

def print_final_table():
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — PLOVER PLV_test (1033 examples)")
    print(f"  Model: {LLM_MODEL} for LLM methods")
    print(f"{'='*70}")

    results = {}
    for name, csv_name in [('ZSP Tree','PLV_test-Tree-result.csv'),
                            ('ZSP Tiny','PLV_test-Tiny-result.csv'),
                            ('ZSP Full','PLV_test-Full-result.csv')]:
        path = f'{REPO}/outputs/{csv_name}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            y_true = df['gold_root'].astype(str).str.upper().tolist()
            best = None
            for s in ['L1','L2']:
                col = f'root_{s}'
                if col in df.columns:
                    y_pred = df[col].astype(str).str.upper().tolist()
                    sc = compute_f1(y_true, y_pred)
                    if best is None or sum(sc)/3 > sum(best)/3:
                        best = sc
            results[name] = best or ('—','—','—')
        else:
            results[name] = ('—','—','—')

    for name, csv_name in [('LLM No Codebook','llm_no_codebook.csv'),
                            ('LLM Compact','llm_compact.csv'),
                            ('LLM CoT','llm_cot.csv'),
                            ('LLM ICL','llm_icl.csv'),('LLM Enriched','llm_enriched.csv')]:
        path = f'{REPO}/outputs/{csv_name}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            y_true = df['gold_root'].astype(str).str.upper().tolist()
            y_pred = df['pred_root'].astype(str).str.upper().tolist()
            results[name] = compute_f1(y_true, y_pred)
        else:
            results[name] = ('—','—','—')

    prior_baselines = {
        '[Prior] ZSP Tree': (96.4, 89.6, 82.4),
        '[Prior] ZSP Tiny': (90.5, 69.5, 50.8),
        '[Prior] ZSP Full': (91.0, 73.4, 55.7),
    }

    print(f"\n{'Method':<28} {'Binary':>8} {'Quad':>8} {'Root':>8} {'Avg':>8}")
    print('─' * 62)
    rows = []
    for method, vals in results.items():
        b, q, r = vals
        if isinstance(b, (int, float)):
            avg = (b+q+r)/3
            print(f"{method:<28} {b:>8.1f} {q:>8.1f} {r:>8.1f} {avg:>8.1f}")
            rows.append({'Method':method,'Binary_F1':round(b,1),'Quad_F1':round(q,1),
                        'Root_F1':round(r,1),'Avg':round(avg,1)})
        else:
            print(f"{method:<28} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
    print('─' * 62)
    for method, vals in prior_baselines.items():
        b, q, r = vals
        avg = (b+q+r)/3
        print(f"{method:<28} {b:>8.1f} {q:>8.1f} {r:>8.1f} {avg:>8.1f}")
        rows.append({'Method':method,'Binary_F1':b,'Quad_F1':q,'Root_F1':r,'Avg':round(avg,1)})

    out = f'{REPO}/outputs/final_results.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved → {out}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='PLOVER Experiments')
    parser.add_argument('--step', required=True,
        choices=['tree','tiny','full','llm_no_cb','llm_compact','llm_enriched','llm_json_cb','llm_cot','llm_icl',
                 'llm_all','nli_all','table','all'])
    parser.add_argument('--limit', type=int, default=None,
        help='Limit LLM experiments to N examples (for quick testing)')
    args = parser.parse_args()

    os.makedirs(f'{REPO}/outputs', exist_ok=True)
    os.makedirs(f'{REPO}/scores', exist_ok=True)

    dispatch = {
        'tree':      lambda: run_zsp_tree(),
        'tiny':      lambda: run_zsp_tiny(),
        'full':      lambda: run_zsp_full(),
        'llm_no_cb': lambda: run_llm_no_codebook(args.limit),
        'llm_compact': lambda: run_llm_compact(args.limit),
        'llm_enriched': lambda: run_llm_enriched(args.limit),
        'llm_json_cb': lambda: run_llm_with_json_codebook(args.limit),
        'llm_cot':   lambda: run_llm_cot(args.limit),
        'llm_icl':   lambda: run_llm_icl(args.limit),
        'nli_all':   lambda: [run_zsp_tree(), run_zsp_tiny(), run_zsp_full()],
        'llm_all':   lambda: [run_llm_no_codebook(args.limit),
                              run_llm_compact(args.limit),
                              run_llm_enriched(args.limit),
                              run_llm_cot(args.limit),
                              run_llm_icl(args.limit)],
        'table':     lambda: print_final_table(),
        'all':       lambda: [run_zsp_tree(), run_zsp_tiny(), run_zsp_full(),
                              run_llm_no_codebook(args.limit),
                              run_llm_compact(args.limit),
                              run_llm_enriched(args.limit),
                              run_llm_cot(args.limit),
                              run_llm_icl(args.limit),
                              print_final_table()],
    }
    dispatch[args.step]()

    if args.step != 'table':
        print(f"\n  → Run 'python3 primary/plover_predictive_experiments.py --step table' for full comparison.")

if __name__ == '__main__':
    main()