#!/usr/bin/env python3
"""
AW_rag_v2.py — Enhanced RAG v2 for A/W binary political event classification

Adapted from the PLV RAG v2 experiment for the A/W dataset (binary: COOPERATION/CONFLICT).

Key features carried over from PLOVER v2:
  1. Flat retrieval with confusable-neighbor forcing
  2. Contrastive example pairs
  3. Upgraded embedding model (all-mpnet-base-v2)
  4. Sentence-first prompt ordering
  5. Expanded example bank with contrastive pairs

Note: Hierarchical retrieval is not applicable for A/W since it is binary-only.
      The 'hier' steps run flat retrieval with contrastive examples instead.

Usage:
    python3 AW_rag_v2.py --step rag_cb
    python3 AW_rag_v2.py --step rag_cb_ex
    python3 AW_rag_v2.py --step rag_noisy
    python3 AW_rag_v2.py --step table
    python3 AW_rag_v2.py --step demo
    python3 AW_rag_v2.py --step rag_cb --limit 5
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
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-mpnet-base-v2')

LABELS = ['COOPERATION', 'CONFLICT']

CODEBOOK_CHUNKS = [
    {"id": "cb_cooperation", "label": "COOPERATION", "type": "definition",
     "text": ("COOPERATION: The source actor engages in cooperative behavior toward "
              "the target. Verbal cooperation includes agreements, consultations, "
              "diplomatic support, promises, meetings, endorsements, treaty signing, "
              "expressions of approval, commendation. Material cooperation includes "
              "providing aid (monetary, military, humanitarian), trade, concessions, "
              "yielding territory, ceasefire, release of prisoners, repatriation, "
              "disarming, easing restrictions, allowing access. "
              "Key signals: agree, consult, support, cooperate, aid, yield, sign, "
              "meet, promise, approve, ease, release, ceasefire, endorse.")},
    {"id": "cb_conflict", "label": "CONFLICT", "type": "definition",
     "text": ("CONFLICT: The source actor engages in conflictual behavior toward "
              "the target. Verbal conflict includes requests, demands, accusations, "
              "rejections, threats, condemnations, complaints, investigations, charges. "
              "Material conflict includes protests, demonstrations, sanctions, reduction "
              "of relations, mobilization of forces, blockades, seizures, arrests, "
              "coercion, armed assault, attacks, bombings, kidnapping, cyber attacks. "
              "Key signals: demand, accuse, reject, threaten, protest, sanction, "
              "mobilize, coerce, assault, attack, arrest, condemn, criticize.")},
]

CB_BY_LABEL = {c['label']: c for c in CODEBOOK_CHUNKS}

DISAMBIGUATION_CHUNKS = [
    {"id": "rule_future_coop", "type": "disambiguation",
     "text": ("FUTURE TENSE RULE: Future-tense cooperative actions ('will provide aid', "
              "'agreed to meet', 'plans to cooperate') are COOPERATION.")},
    {"id": "rule_negated_coop", "type": "disambiguation",
     "text": ("NEGATED COOPERATION RULE: Halting, suspending, or cutting off existing "
              "cooperation ('halted aid', 'suspended trade', 'withdrew ambassador') "
              "is CONFLICT because it reduces cooperative relations.")},
    {"id": "rule_peacekeeping", "type": "disambiguation",
     "text": ("PEACEKEEPING RULE: Peacekeeping forces, humanitarian workers, and "
              "observers indicate COOPERATION even when involving military actors.")},
    {"id": "rule_demands", "type": "disambiguation",
     "text": ("DEMANDS RULE: Demands, requests, and orders are CONFLICT even when "
              "they seek cooperative outcomes (e.g. 'demanded a ceasefire').")},
    {"id": "rule_legal", "type": "disambiguation",
     "text": ("LEGAL PROCEEDINGS RULE: Investigations, charges, lawsuits, arrests, "
              "and legal proceedings are CONFLICT.")},
    {"id": "rule_protests", "type": "disambiguation",
     "text": ("PROTESTS RULE: Planned or actual protests and demonstrations are CONFLICT.")},
    {"id": "rule_concessions", "type": "disambiguation",
     "text": ("CONCESSIONS RULE: Yielding, concessions, retreats, prisoner releases, "
              "and easing of restrictions are COOPERATION.")},
    {"id": "rule_sanctions", "type": "disambiguation",
     "text": ("SANCTIONS RULE: Imposing sanctions, embargoes, trade restrictions, "
              "or blockades is CONFLICT. Lifting them is COOPERATION.")},
    {"id": "rule_symbolic", "type": "disambiguation",
     "text": ("SYMBOLIC THREATS RULE: Symbolic threats, shows of force, and military "
              "posturing are CONFLICT even without actual violence.")},
    {"id": "rule_signing", "type": "disambiguation",
     "text": ("SIGNING RULE: Signing, ratifying, or finalizing agreements and treaties "
              "is COOPERATION.")},
]

EXAMPLE_BANK = [
    {"text": "<S>The president</S> agreed to sign the peace treaty with <T>the opposition</T>.",
     "label": "COOPERATION", "explanation": "Verbal commitment to cooperate."},
    {"text": "<S>Delegates</S> held talks with <T>foreign ministers</T> at the UN summit.",
     "label": "COOPERATION", "explanation": "Meeting/consultation between parties."},
    {"text": "<S>The government</S> praised <T>the international community</T> for its efforts.",
     "label": "COOPERATION", "explanation": "Verbal endorsement."},
    {"text": "<S>Both countries</S> signed a mutual defense pact with <T>each other</T>.",
     "label": "COOPERATION", "explanation": "Bilateral material cooperation."},
    {"text": "<S>The UN</S> delivered humanitarian supplies to <T>the refugees</T>.",
     "label": "COOPERATION", "explanation": "Unidirectional material aid."},
    {"text": "<S>The military</S> released political prisoners held by <T>the regime</T>.",
     "label": "COOPERATION", "explanation": "Material concession."},
    {"text": "<S>The ambassador</S> demanded <T>the foreign government</T> release the hostages.",
     "label": "CONFLICT", "explanation": "Verbal demand."},
    {"text": "<S>Opposition leaders</S> accused <T>the ruling party</T> of corruption.",
     "label": "CONFLICT", "explanation": "Verbal blame directed at target."},
    {"text": "<S>The parliament</S> rejected <T>the president's</T> proposed budget.",
     "label": "CONFLICT", "explanation": "Explicit refusal."},
    {"text": "<S>The general</S> warned <T>neighboring forces</T> of severe consequences.",
     "label": "CONFLICT", "explanation": "Verbal warning of punitive action."},
    {"text": "<S>Thousands of citizens</S> marched against <T>the government</T>.",
     "label": "CONFLICT", "explanation": "Collective physical opposition."},
    {"text": "<S>The EU</S> withdrew its ambassador from <T>the country</T>.",
     "label": "CONFLICT", "explanation": "Reduction of diplomatic relations."},
    {"text": "<S>The army</S> deployed troops along the border with <T>the neighbor</T>.",
     "label": "CONFLICT", "explanation": "Military positioning."},
    {"text": "<S>Police</S> arrested dozens of <T>opposition activists</T>.",
     "label": "CONFLICT", "explanation": "Non-violent coercive action."},
    {"text": "<S>Rebel forces</S> launched an attack on <T>government positions</T>.",
     "label": "CONFLICT", "explanation": "Actual use of force."},
    {"text": "<S>The president</S> will provide economic assistance to <T>the ally</T>.",
     "label": "COOPERATION", "explanation": "CONTRASTIVE: Future tense cooperative -> COOPERATION."},
    {"text": "<S>The government</S> halted economic assistance to <T>the ally</T>.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Negated cooperation -> CONFLICT."},
    {"text": "<S>The foreign minister</S> plans to meet with <T>the delegation</T> next week.",
     "label": "COOPERATION", "explanation": "CONTRASTIVE: Future meeting -> COOPERATION."},
    {"text": "<S>The foreign minister</S> refused to meet with <T>the delegation</T>.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Refusal to meet -> CONFLICT."},
    {"text": "<S>The government</S> promised to ease restrictions on <T>the opposition</T>.",
     "label": "COOPERATION", "explanation": "CONTRASTIVE: Promise to yield -> COOPERATION."},
    {"text": "<S>The government</S> tightened restrictions on <T>the opposition</T>.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Increasing restrictions -> CONFLICT."},
    {"text": "<S>Protesters</S> demanded reforms from <T>the government</T> at a rally.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Demand via demonstration -> CONFLICT."},
    {"text": "<S>The ambassador</S> called on <T>the government</T> to implement reforms.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Verbal demand -> CONFLICT."},
    {"text": "<S>Authorities</S> convicted and imprisoned <T>the dissident</T>.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Coercive material action -> CONFLICT."},
    {"text": "<S>Security forces</S> detained <T>journalists</T> covering the protests.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Detention -> CONFLICT."},
    {"text": "<S>Security forces</S> fired on <T>protesters</T> with live ammunition.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Physical violence -> CONFLICT."},
    {"text": "<S>The two nations</S> launched a joint intelligence-sharing program with <T>each other</T>.",
     "label": "COOPERATION", "explanation": "CONTRASTIVE: Mutual material exchange -> COOPERATION."},
    {"text": "<S>The donor country</S> shipped medical supplies to <T>the disaster zone</T>.",
     "label": "COOPERATION", "explanation": "CONTRASTIVE: One-way material aid -> COOPERATION."},
    {"text": "<S>The country</S> imposed trade restrictions on <T>the rival state</T>.",
     "label": "CONFLICT", "explanation": "CONTRASTIVE: Punitive trade action -> CONFLICT."},
    {"text": "<S>The country</S> lifted trade restrictions on <T>the partner state</T>.",
     "label": "COOPERATION", "explanation": "CONTRASTIVE: Lifting restrictions -> COOPERATION."},
    {"text": "<S>The regime</S> shut down newspapers operated by <T>the opposition</T>.",
     "label": "CONFLICT", "explanation": "Restricting press freedom -> CONFLICT."},
    {"text": "<S>Armed groups</S> kidnapped <T>aid workers</T> in the region.",
     "label": "CONFLICT", "explanation": "Kidnapping = physical harm -> CONFLICT."},
    {"text": "<S>The government</S> signed the ceasefire agreement with <T>rebel forces</T>.",
     "label": "COOPERATION", "explanation": "Signing agreement -> COOPERATION."},
    {"text": "<S>Military forces</S> retreated from <T>the occupied zone</T>.",
     "label": "COOPERATION", "explanation": "Military retreat = concession -> COOPERATION."},
    {"text": "<S>The navy</S> imposed a blockade on <T>the port city</T>.",
     "label": "CONFLICT", "explanation": "Blockade = material conflict -> CONFLICT."},
    {"text": "<S>Aid workers</S> distributed food and medicine to <T>refugees</T>.",
     "label": "COOPERATION", "explanation": "Humanitarian aid -> COOPERATION."},
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
    return bool(re.search(r'<S>.*?</S>', text) and re.search(r'<T>.*?</T>', text))


def add_noise_context_to_prompt(text):
    if not has_source_target_markers(text):
        return ("\nNOTE: This sentence may be missing <S>/<T> actor markers. "
                "Identify the most likely source (initiator of the action) and "
                "target (receiver) from context, then classify the relation.")
    return ""


class AWRagV2:
    def __init__(self, embed_model=EMBED_MODEL,
                 top_k_codebook=2, top_k_rules=3, top_k_examples=4,
                 use_noise_preprocessing=False):
        print(f"  Loading embedding model: {embed_model} ...")
        self.embedder = SentenceTransformer(embed_model)
        self.top_k_codebook = top_k_codebook
        self.top_k_rules = top_k_rules
        self.top_k_examples = top_k_examples
        self.use_noise_preprocessing = use_noise_preprocessing
        print("  Building FAISS indices ...")
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
        print("  RAG v2 engine ready.\n")

    def retrieve_flat(self, sentence, include_examples=False):
        if self.use_noise_preprocessing:
            sentence = preprocess_noisy_text(sentence)
        q_emb = self.embedder.encode([sentence], convert_to_numpy=True,
                                      normalize_embeddings=True).astype('float32')
        _, cb_idx = self.cb_index.search(q_emb, min(self.top_k_codebook, len(self.cb_texts)))
        _, rule_idx = self.rule_index.search(q_emb, min(self.top_k_rules, len(self.rule_texts)))
        codebook_chunks = [self.cb_meta[i] for i in cb_idx[0]]
        retrieved_labels = set(c['label'] for c in codebook_chunks)
        for label in LABELS:
            if label not in retrieved_labels:
                codebook_chunks.append(CB_BY_LABEL[label])
        result = {
            'codebook_chunks': codebook_chunks,
            'rules': [self.rule_meta[i] for i in rule_idx[0]],
            'examples': [],
            'noise_note': '',
        }
        if include_examples:
            _, ex_idx = self.ex_index.search(q_emb, min(self.top_k_examples, len(self.ex_texts)))
            result['examples'] = [self.ex_meta[i] for i in ex_idx[0]]
        return result

    def build_prompt(self, sentence, strategy='cb'):
        clean = preprocess_noisy_text(sentence) if (
            self.use_noise_preprocessing or strategy == 'noisy') else sentence
        include_ex = strategy in ('cb_ex', 'noisy')
        retrieved = self.retrieve_flat(clean, include_examples=include_ex)
        parts = []
        parts.append("You are a political event classifier.")
        parts.append(f"\nSENTENCE TO CLASSIFY:\n{clean}")
        parts.append("\nRELEVANT LABEL DEFINITIONS:")
        seen = set()
        for i, chunk in enumerate(retrieved['codebook_chunks'], 1):
            if chunk['label'] not in seen:
                parts.append(f"{i}. {chunk['text']}")
                seen.add(chunk['label'])
        if retrieved['rules']:
            parts.append("\nDISAMBIGUATION RULES:")
            for rule in retrieved['rules']:
                parts.append(f"- {rule['text']}")
        if retrieved['examples']:
            parts.append("\nSIMILAR LABELED EXAMPLES:")
            for ex in retrieved['examples']:
                parts.append(f"  Input: {ex['text']}")
                parts.append(f"  Label: {ex['label']} ({ex['explanation']})")
                parts.append("")
        if strategy == 'noisy':
            noise_note = add_noise_context_to_prompt(clean)
            if noise_note:
                parts.append(noise_note)
        parts.append(f"\nVALID LABELS: COOPERATION, CONFLICT")
        parts.append("\nOutput ONLY the label name (COOPERATION or CONFLICT), nothing else.")
        return '\n'.join(parts)


def query_ollama(prompt, model=None, temperature=0.1, max_retries=3):
    if model is None:
        model = LLM_MODEL
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 20}
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get('response', '').strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Ollama error after {max_retries} attempts: {e}")
                return ""


def extract_label(text):
    if not text:
        return "UNKNOWN"
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
    return "UNKNOWN"


def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0) * 100


def load_test_data(limit=None):
    path = f'{REPO}/datasets/AW_test.tsv'
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
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


STRATEGY_MAP = {
    'rag_cb':    'cb',
    'rag_cb_ex': 'cb_ex',
    'rag_noisy': 'noisy',
}


def run_rag_experiment(name, strategy, save_path, limit=None, noise_preprocess=False):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Strategy: {strategy} | Model: {LLM_MODEL}")
    print(f"  Embed model: {EMBED_MODEL}")
    print(f"{'='*70}\n")
    rag = AWRagV2(use_noise_preprocessing=noise_preprocess)
    df = load_test_data(limit)
    sent_col = 'marked_sentence' if 'marked_sentence' in df.columns else 'sentence'
    truths, preds = [], []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        sentence = str(row[sent_col])
        gold = gold_to_label(row['gold_binary'])
        prompt = rag.build_prompt(sentence, strategy=strategy)
        response = query_ollama(prompt)
        pred = extract_label(response)
        if pred == 'UNKNOWN':
            pred = 'CONFLICT'
        truths.append(gold)
        preds.append(pred)
        if (idx + 1) % 100 == 0 or idx == 0:
            f1 = compute_f1(truths, preds)
            print(f"  [{idx+1:>4}/{total}]  Binary F1={f1:.1f}  last_pred={pred}  gold={gold}")
    f1 = compute_f1(truths, preds)
    print(f"\n  FINAL — Binary F1={f1:.1f}")
    print(f"  Baselines: UP=67.2  ZSP Tree=88.0")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df = pd.DataFrame({
        'sentence': [str(row[sent_col]) for _, row in load_test_data(limit).iterrows()],
        'gold': truths,
        'pred': preds,
    })
    results_df.to_csv(save_path, index=False)
    print(f"  Saved to: {save_path}")
    return f1


def print_comparison_table():
    print(f"\n{'='*60}")
    print("  A/W Binary Classification — RAG v2 Comparison Table")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'Binary F1':>10}")
    print("  " + "-"*40)
    baselines = [
        ('Prior UP', 67.2),
        ('Prior ZSP Tree', 88.0),
    ]
    for name, f1 in baselines:
        print(f"  {name:<30} {f1:>10.1f}")
    print("  " + "-"*40)
    output_dir = f'{REPO}/outputs'
    rag_files = {
        'RAG v2 Flat (CB)':       'aw_rag_v2_cb.csv',
        'RAG v2 Flat (CB+Ex)':    'aw_rag_v2_cb_ex.csv',
        'RAG v2 Noisy':           'aw_rag_v2_noisy.csv',
        'RAG v1 Codebook':        'aw_rag_codebook.csv',
        'RAG v1 CB+Examples':     'aw_rag_cb_examples.csv',
        'RAG v1 Noisy':           'aw_rag_noisy.csv',
    }
    for name, fname in rag_files.items():
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            f1 = compute_f1(df['gold'].tolist(), df['pred'].tolist())
            print(f"  {name:<30} {f1:>10.1f}")
        else:
            print(f"  {name:<30} {'—':>10}")
    print("  " + "-"*40)
    print()


def main():
    global LLM_MODEL, EMBED_MODEL
    parser = argparse.ArgumentParser(description="A/W RAG v2 Classification Module")
    parser.add_argument('--step', required=True,
                        choices=['rag_cb', 'rag_cb_ex', 'rag_noisy', 'table', 'demo'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model', type=str, default=LLM_MODEL)
    parser.add_argument('--embed', type=str, default=EMBED_MODEL)
    args = parser.parse_args()
    LLM_MODEL = args.model
    EMBED_MODEL = args.embed
    output_dir = f'{REPO}/outputs'
    os.makedirs(output_dir, exist_ok=True)
    strategy = STRATEGY_MAP.get(args.step, args.step)
    if args.step == 'table':
        print_comparison_table()
    elif args.step == 'demo':
        print("\n  Interactive AW RAG v2 Demo (type 'quit' to exit)\n")
        rag = AWRagV2(embed_model=EMBED_MODEL)
        while True:
            sentence = input("\n  Enter sentence: ").strip()
            if sentence.lower() in ('quit', 'exit', 'q'):
                break
            for s in ['cb', 'cb_ex']:
                prompt = rag.build_prompt(sentence, strategy=s)
                response = query_ollama(prompt)
                pred = extract_label(response)
                print(f"  [{s:>8}] -> {pred}  (prompt: {len(prompt)} chars)")
    else:
        step_to_filename = {
            'rag_cb':    'aw_rag_v2_cb.csv',
            'rag_cb_ex': 'aw_rag_v2_cb_ex.csv',
            'rag_noisy': 'aw_rag_v2_noisy.csv',
        }
        fname = step_to_filename[args.step]
        noise = args.step == 'rag_noisy'
        run_rag_experiment(
            name=f"AW RAG v2 — {args.step}",
            strategy=strategy,
            save_path=f'{output_dir}/{fname}',
            limit=args.limit,
            noise_preprocess=noise
        )
    print("\nDone.")


if __name__ == '__main__':
    main()