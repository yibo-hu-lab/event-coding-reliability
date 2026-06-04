#!/usr/bin/env python3
"""
aw_rag_experiments.py — RAG module for A/W binary political event classification

Adapted from PLOVER_rag.py for the A/W dataset (binary: COOPERATION/CONFLICT).

Usage:
    python3 supplemental/rag/aw_rag_experiments.py --step rag_cb
    python3 supplemental/rag/aw_rag_experiments.py --step rag_cb_ex
    python3 supplemental/rag/aw_rag_experiments.py --step rag_noisy
    python3 supplemental/rag/aw_rag_experiments.py --step table
    python3 supplemental/rag/aw_rag_experiments.py --step demo
    python3 supplemental/rag/aw_rag_experiments.py --step rag_cb --limit 5
"""

import os, sys, re, time, argparse
from pathlib import Path
import pandas as pd
import requests
from sklearn.metrics import f1_score

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install sentence-transformers faiss-cpu")
    sys.exit(1)

REPO       = os.environ.get('PLOVER_REPO', str(Path(__file__).resolve().parents[2]))
LLM_MODEL  = os.environ.get('PLOVER_LLM', 'gemma2:9b')
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')

LABELS = ['COOPERATION', 'CONFLICT']

CODEBOOK_CHUNKS = [
    {
        "id": "cb_cooperation",
        "label": "COOPERATION",
        "type": "definition",
        "text": ("COOPERATION: The source actor engages in cooperative behavior toward "
                 "the target. This includes verbal cooperation (agreements, consultations, "
                 "diplomatic support, promises, meetings, endorsements, treaty signing) "
                 "AND material cooperation (providing aid, trade, concessions, yielding "
                 "territory, ceasefire, release of prisoners, repatriation, disarming, "
                 "easing restrictions). Key signals: agree, consult, support, cooperate, "
                 "aid, yield, sign, meet, promise, approve, ease, release, ceasefire.")
    },
    {
        "id": "cb_conflict",
        "label": "CONFLICT",
        "type": "definition",
        "text": ("CONFLICT: The source actor engages in conflictual behavior toward "
                 "the target. This includes verbal conflict (requests, demands, accusations, "
                 "rejections, threats, condemnations, complaints, investigations, charges) "
                 "AND material conflict (protests, sanctions, mobilization of forces, "
                 "blockades, seizures, arrests, coercion, armed assault, attacks, "
                 "bombings, kidnapping). Key signals: demand, accuse, reject, threaten, "
                 "protest, sanction, mobilize, coerce, assault, attack, arrest, condemn.")
    },
]

DISAMBIGUATION_CHUNKS = [
    {
        "id": "rule_future_coop",
        "type": "disambiguation",
        "text": ("FUTURE TENSE RULE: Future-tense cooperative actions ('will provide aid', "
                 "'agreed to meet', 'plans to cooperate') are COOPERATION, not CONFLICT.")
    },
    {
        "id": "rule_negated_coop",
        "type": "disambiguation",
        "text": ("NEGATED COOPERATION RULE: Halting, suspending, or cutting off existing "
                 "cooperation ('halted aid', 'suspended trade', 'withdrew ambassador') "
                 "is CONFLICT because it reduces cooperative relations.")
    },
    {
        "id": "rule_peacekeeping",
        "type": "disambiguation",
        "text": ("PEACEKEEPING RULE: Peacekeeping forces, humanitarian workers, and "
                 "observers indicate COOPERATION even though they involve military or "
                 "security actors.")
    },
    {
        "id": "rule_demands",
        "type": "disambiguation",
        "text": ("DEMANDS RULE: Demands, requests, and orders are CONFLICT even when "
                 "they seek cooperative outcomes (e.g. 'demanded a ceasefire'). The "
                 "coercive nature of the demand overrides the cooperative goal.")
    },
    {
        "id": "rule_legal",
        "type": "disambiguation",
        "text": ("LEGAL PROCEEDINGS RULE: Investigations, charges, lawsuits, arrests, "
                 "and legal proceedings are CONFLICT (accusations/coercion against target).")
    },
    {
        "id": "rule_protests",
        "type": "disambiguation",
        "text": ("PROTESTS RULE: Planned or actual protests and demonstrations are "
                 "CONFLICT regardless of whether they have occurred yet.")
    },
    {
        "id": "rule_concessions",
        "type": "disambiguation",
        "text": ("CONCESSIONS RULE: Yielding, concessions, retreats, prisoner releases, "
                 "and easing of restrictions are COOPERATION because they reduce conflict.")
    },
    {
        "id": "rule_sanctions",
        "type": "disambiguation",
        "text": ("SANCTIONS RULE: Imposing sanctions, embargoes, trade restrictions, "
                 "or blockades is CONFLICT. Lifting them is COOPERATION.")
    },
    {
        "id": "rule_symbolic_threats",
        "type": "disambiguation",
        "text": ("SYMBOLIC THREATS RULE: Symbolic threats, shows of force, and military "
                 "posturing are CONFLICT even without actual violence.")
    },
]

EXAMPLE_BANK = [
    {"text": "<S>The president</S> agreed to sign the peace treaty with <T>the opposition</T>.", "label": "COOPERATION", "explanation": "Verbal commitment to cooperate (sign treaty)."},
    {"text": "<S>Delegates</S> held talks with <T>foreign ministers</T> at the UN summit.", "label": "COOPERATION", "explanation": "Meeting/consultation between parties."},
    {"text": "<S>The government</S> praised <T>the international community</T> for its efforts.", "label": "COOPERATION", "explanation": "Verbal endorsement."},
    {"text": "<S>Both countries</S> signed a mutual defense pact with <T>each other</T>.", "label": "COOPERATION", "explanation": "Bilateral material cooperation."},
    {"text": "<S>The UN</S> delivered humanitarian supplies to <T>the refugees</T>.", "label": "COOPERATION", "explanation": "Unidirectional material aid."},
    {"text": "<S>The military</S> released political prisoners held by <T>the regime</T>.", "label": "COOPERATION", "explanation": "Material concession (releasing prisoners)."},
    {"text": "<S>The government</S> eased visa restrictions for <T>allied citizens</T>.", "label": "COOPERATION", "explanation": "Easing restrictions = concession."},
    {"text": "<S>The two leaders</S> consulted by phone regarding <T>the border dispute</T>.", "label": "COOPERATION", "explanation": "Consultation between parties."},
    {"text": "<S>The ambassador</S> condemned <T>the military strikes</T> on civilian targets.", "label": "CONFLICT", "explanation": "Verbal blame/condemnation."},
    {"text": "<S>Protesters</S> gathered outside <T>the embassy</T> demanding the release of prisoners.", "label": "CONFLICT", "explanation": "Collective physical opposition."},
    {"text": "<S>The government</S> imposed economic sanctions on <T>the rival nation</T>.", "label": "CONFLICT", "explanation": "Punitive reduction in relations."},
    {"text": "<S>Armed forces</S> launched an assault on <T>rebel-held territory</T>.", "label": "CONFLICT", "explanation": "Actual physical violence."},
    {"text": "<S>The president</S> rejected <T>the proposed peace deal</T>.", "label": "CONFLICT", "explanation": "Explicit refusal of cooperation."},
    {"text": "<S>Security forces</S> arrested several <T>opposition leaders</T>.", "label": "CONFLICT", "explanation": "Non-violent coercive action."},
    {"text": "<S>The country</S> threatened to withdraw from <T>the international agreement</T>.", "label": "CONFLICT", "explanation": "Verbal warning of punitive action."},
    {"text": "<S>Troops</S> were mobilized along the disputed border with <T>the neighbor</T>.", "label": "CONFLICT", "explanation": "Military positioning."},
    {"text": "<S>The court</S> charged <T>the former official</T> with crimes against humanity.", "label": "CONFLICT", "explanation": "Legal accusation."},
    {"text": "<S>The government</S> halted all foreign aid to <T>the region</T>.", "label": "CONFLICT", "explanation": "Negated cooperation = conflict."},
    {"text": "<S>The alliance</S> agreed to provide military equipment to <T>the besieged city</T>.", "label": "COOPERATION", "explanation": "Commitment to provide aid."},
    {"text": "<S>The opposition</S> demanded immediate elections and threatened <T>mass protests</T>.", "label": "CONFLICT", "explanation": "Demands + threat of collective action."},
]


def preprocess_noisy_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<(?!/?[ST]>)[^>]+>', '', text)
    replacements = {
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\xa0': ' ',
        '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 2 and text[0] in '""\'' and text[-1] in '""\'':
        text = text[1:-1].strip()
    return text


def has_source_target_markers(text):
    return bool(re.search(r'<S>', text) and re.search(r'<T>', text))


def add_noise_context_to_prompt(text):
    if has_source_target_markers(text):
        return ""
    return ("\nNOTE: This sentence lacks explicit <S>source</S> and <T>target</T> "
            "markers. Identify the primary political actors and their roles from "
            "context. The source is the actor performing the action; the target "
            "is the actor receiving it.\n")


class AWRag:
    def __init__(self, embed_model_name=EMBED_MODEL,
                 top_k_codebook=2, top_k_rules=2, top_k_examples=3,
                 use_noise_preprocessing=False, verbose=True):
        self.top_k_codebook = top_k_codebook
        self.top_k_rules = top_k_rules
        self.top_k_examples = top_k_examples
        self.use_noise_preprocessing = use_noise_preprocessing
        self.verbose = verbose
        if verbose:
            print(f"  Loading embedding model: {embed_model_name} ...")
        self.embedder = SentenceTransformer(embed_model_name)
        self._build_indices()
        if verbose:
            print(f"  RAG ready: {len(self.cb_texts)} codebook chunks, "
                  f"{len(self.rule_texts)} rules, {len(self.ex_texts)} examples")

    def _build_indices(self):
        self.cb_texts = [c['text'] for c in CODEBOOK_CHUNKS]
        self.cb_meta  = CODEBOOK_CHUNKS
        cb_embs = self.embedder.encode(self.cb_texts, convert_to_numpy=True,
                                        normalize_embeddings=True)
        self.cb_index = faiss.IndexFlatIP(cb_embs.shape[1])
        self.cb_index.add(cb_embs.astype('float32'))

        self.rule_texts = [r['text'] for r in DISAMBIGUATION_CHUNKS]
        self.rule_meta  = DISAMBIGUATION_CHUNKS
        rule_embs = self.embedder.encode(self.rule_texts, convert_to_numpy=True,
                                          normalize_embeddings=True)
        self.rule_index = faiss.IndexFlatIP(rule_embs.shape[1])
        self.rule_index.add(rule_embs.astype('float32'))

        self.ex_texts = [e['text'] for e in EXAMPLE_BANK]
        self.ex_meta  = EXAMPLE_BANK
        ex_embs = self.embedder.encode(self.ex_texts, convert_to_numpy=True,
                                        normalize_embeddings=True)
        self.ex_index = faiss.IndexFlatIP(ex_embs.shape[1])
        self.ex_index.add(ex_embs.astype('float32'))

    def retrieve(self, sentence, strategy='cb'):
        if self.use_noise_preprocessing or strategy == 'noisy':
            sentence = preprocess_noisy_text(sentence)
        q_emb = self.embedder.encode([sentence], convert_to_numpy=True,
                                      normalize_embeddings=True).astype('float32')
        _, cb_idx = self.cb_index.search(q_emb, min(self.top_k_codebook, len(self.cb_texts)))
        _, rule_idx = self.rule_index.search(q_emb, min(self.top_k_rules, len(self.rule_texts)))
        result = {
            'codebook_chunks': [self.cb_meta[i] for i in cb_idx[0]],
            'rules': [self.rule_meta[i] for i in rule_idx[0]],
            'examples': [],
            'noise_note': '',
        }
        if strategy in ('cb_ex', 'noisy'):
            _, ex_idx = self.ex_index.search(q_emb, min(self.top_k_examples, len(self.ex_texts)))
            result['examples'] = [self.ex_meta[i] for i in ex_idx[0]]
        if strategy == 'noisy':
            result['noise_note'] = add_noise_context_to_prompt(sentence)
        return result

    def build_prompt(self, sentence, strategy='cb'):
        if self.use_noise_preprocessing or strategy == 'noisy':
            clean_sentence = preprocess_noisy_text(sentence)
        else:
            clean_sentence = sentence
        retrieved = self.retrieve(sentence, strategy)
        parts = []
        parts.append("You are a political event classifier. "
                      "Classify the relation between source (<S></S>) and target (<T></T>).")
        parts.append("\nRELEVANT LABEL DEFINITIONS:")
        for i, chunk in enumerate(retrieved['codebook_chunks'], 1):
            parts.append(f"{i}. {chunk['text']}")
        if retrieved['rules']:
            parts.append("\nDISAMBIGUATION RULES (apply these when choosing between labels):")
            for rule in retrieved['rules']:
                parts.append(f"- {rule['text']}")
        if retrieved['examples']:
            parts.append("\nSIMILAR LABELED EXAMPLES:")
            for ex in retrieved['examples']:
                parts.append(f"  Sentence: {ex['text']}")
                parts.append(f"  Label: {ex['label']} — {ex['explanation']}")
                parts.append("")
        if retrieved['noise_note']:
            parts.append(retrieved['noise_note'])
        parts.append(f"\nVALID LABELS: COOPERATION, CONFLICT")
        parts.append(f"\nSentence: {clean_sentence}")
        parts.append("\nOutput ONLY the label name (COOPERATION or CONFLICT), nothing else.")
        return '\n'.join(parts)


def query_ollama(prompt, model=LLM_MODEL, temperature=0.1, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                'model': model, 'prompt': prompt, 'stream': False,
                'options': {'temperature': temperature, 'num_predict': 20}
            }, timeout=120)
            resp.raise_for_status()
            return resp.json().get('response', '').strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Ollama error (attempt {attempt+1}): {e}")
                time.sleep(5)
            else:
                print(f"    Ollama failed after {max_retries} attempts: {e}")
                return ""
    return ""


def extract_label(text):
    if not text:
        return 'UNKNOWN'
    text_upper = text.upper().strip()
    aliases = {
        'COOPERATIVE': 'COOPERATION', 'COOP': 'COOPERATION',
        'COOPERATE': 'COOPERATION', 'PEACE': 'COOPERATION',
        'CONFLICTUAL': 'CONFLICT', 'CONF': 'CONFLICT',
        'HOSTILE': 'CONFLICT', 'VIOLENCE': 'CONFLICT',
        'AGREE': 'COOPERATION', 'CONSULT': 'COOPERATION', 'SUPPORT': 'COOPERATION',
        'AID': 'COOPERATION', 'YIELD': 'COOPERATION',
        'REQUEST': 'CONFLICT', 'ACCUSE': 'CONFLICT', 'REJECT': 'CONFLICT',
        'THREATEN': 'CONFLICT', 'PROTEST': 'CONFLICT', 'SANCTION': 'CONFLICT',
        'MOBILIZE': 'CONFLICT', 'COERCE': 'CONFLICT', 'ASSAULT': 'CONFLICT',
    }
    for label in LABELS:
        if text_upper == label or text_upper.startswith(label):
            return label
    for label in LABELS:
        if label in text_upper:
            return label
    for alias, mapped in aliases.items():
        if alias in text_upper:
            return mapped
    return 'UNKNOWN'


def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0) * 100


def load_test_data(limit=None):
    path = f'{REPO}/datasets/AW_test.tsv'
    if not os.path.exists(path):
        print(f"  ERROR: Test data not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path, sep='\t')
    valid_prefixes = ["Life","Conflict","Justice","Personnel","Contact","Transaction","Business"]
    df = df[df["event_type"].apply(lambda et: any(et.startswith(p) for p in valid_prefixes))]
    df = df.reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def gold_to_label(val):
    return 'COOPERATION' if int(val) == 1 else 'CONFLICT'


def run_rag_experiment(name, strategy, save_path, limit=None, noise_preprocess=False):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Strategy: {strategy} | Model: {LLM_MODEL}")
    print(f"{'='*60}")
    rag = AWRag(use_noise_preprocessing=noise_preprocess)
    df = load_test_data(limit)
    sent_col = 'marked_sentence' if 'marked_sentence' in df.columns else 'sentence'
    total = len(df)
    preds, truths, errors = [], [], 0
    start_time = time.time()
    print(f"  Running on {total} examples...\n")
    for idx, row in df.iterrows():
        sentence = str(row[sent_col])
        true_label = gold_to_label(row['gold_binary'])
        prompt = rag.build_prompt(sentence, strategy=strategy)
        response = query_ollama(prompt)
        pred = extract_label(response)
        if pred == 'UNKNOWN':
            errors += 1
            pred = 'CONFLICT'
        preds.append(pred)
        truths.append(true_label)
        done = len(preds)
        if done % 100 == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"    [{done}/{total}] {rate:.1f} ex/s | "
                  f"ETA: {eta/60:.1f}min | Errors: {errors}")
    f1 = compute_f1(truths, preds)
    elapsed = time.time() - start_time
    print(f"\n  RESULTS ({name}):")
    print(f"    Binary F1: {f1:.1f}")
    print(f"    Errors:    {errors}/{total}")
    print(f"    Time:      {elapsed/60:.1f} min")
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    results_df = pd.DataFrame({
        'sentence': [str(row[sent_col]) for _, row in load_test_data(limit).iterrows()],
        'gold': truths,
        'pred': preds,
    })
    results_df.to_csv(save_path, index=False)
    print(f"    Saved to: {save_path}")
    return f1


def print_comparison_table():
    print("\n" + "="*55)
    print("  A/W Binary Classification — RAG v1 Comparison Table")
    print("="*55)
    print(f"  {'Method':<25} {'Binary F1':>10}")
    print("  " + "-"*35)
    output_dir = f'{REPO}/outputs'
    rag_files = {
        'RAG Codebook': 'aw_rag_codebook.csv',
        'RAG CB + Examples': 'aw_rag_cb_examples.csv',
        'RAG Noisy': 'aw_rag_noisy.csv',
    }
    for name, fname in rag_files.items():
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            f1 = compute_f1(df['gold'].tolist(), df['pred'].tolist())
            print(f"  {name:<25} {f1:>10.1f}")
        else:
            print(f"  {name:<25} {'—':>10}")
    print("  " + "-"*35)
    print("  Prior UP                 67.2")
    print("  Prior ZSP Tree           88.0")
    print()


def main():
    global LLM_MODEL
    parser = argparse.ArgumentParser(description="A/W RAG v1 Classification Module")
    parser.add_argument('--step', required=True,
                        choices=['rag_cb', 'rag_cb_ex', 'rag_noisy', 'table', 'demo'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model', type=str, default=LLM_MODEL)
    args = parser.parse_args()
    LLM_MODEL = args.model
    output_dir = f'{REPO}/outputs'
    os.makedirs(output_dir, exist_ok=True)
    if args.step == 'rag_cb':
        run_rag_experiment("AW RAG Codebook Only", 'cb',
                           f'{output_dir}/aw_rag_codebook.csv', limit=args.limit)
    elif args.step == 'rag_cb_ex':
        run_rag_experiment("AW RAG Codebook + Examples", 'cb_ex',
                           f'{output_dir}/aw_rag_cb_examples.csv', limit=args.limit)
    elif args.step == 'rag_noisy':
        run_rag_experiment("AW RAG Noisy (CB + Ex + Preprocessing)", 'noisy',
                           f'{output_dir}/aw_rag_noisy.csv', limit=args.limit,
                           noise_preprocess=True)
    elif args.step == 'table':
        print_comparison_table()
    elif args.step == 'demo':
        print("\n  Interactive AW RAG Demo (type 'quit' to exit)\n")
        rag = AWRag()
        while True:
            sentence = input("\n  Enter sentence: ").strip()
            if sentence.lower() in ('quit', 'exit', 'q'):
                break
            prompt = rag.build_prompt(sentence, strategy='cb_ex')
            print(f"\n  --- Generated Prompt ({len(prompt)} chars) ---")
            print(prompt)
            print(f"\n  --- Querying {LLM_MODEL} ---")
            response = query_ollama(prompt)
            pred = extract_label(response)
            print(f"  Raw response: {response}")
            print(f"  Predicted label: {pred}")
    print("\nDone.")


if __name__ == '__main__':
    main()