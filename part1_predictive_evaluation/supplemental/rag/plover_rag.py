#!/usr/bin/env python3
"""
plover_rag.py — RAG module for PLOVER political event classification

Retrieval-Augmented Generation layer that dynamically selects the most
relevant codebook entries, disambiguation rules, and (optionally) similar
labeled examples for each input sentence before sending to the LLM.

Designed to plug into the PLV predictive experiment script.
with Ollama serving gemma2:9b or qwen2.5:7b.

Requirements:
    pip install sentence-transformers faiss-cpu pandas scikit-learn

Usage as standalone:
    python3 supplemental/rag/plover_rag.py --step rag_cb          # RAG + codebook chunks
    python3 supplemental/rag/plover_rag.py --step rag_cb_ex        # RAG + codebook + example retrieval
    python3 supplemental/rag/plover_rag.py --step rag_noisy        # RAG with noise-robust preprocessing
    python3 supplemental/rag/plover_rag.py --step table            # Print comparison table
    python3 supplemental/rag/plover_rag.py --step rag_cb --limit 5 # Quick test on 5 examples

Usage as importable module from `plover_predictive_experiments.py`:
    from plover_rag import PLOVERRag
    rag = PLOVERRag()
    prompt = rag.build_prompt(sentence, strategy='cb_ex')
"""

import os, sys, re, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Try importing ML deps; provide clear error if missing
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install sentence-transformers faiss-cpu")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants mirrored from the PLV predictive experiment script for standalone use.
# ---------------------------------------------------------------------------
REPO       = os.environ.get('PLOVER_REPO', str(Path(__file__).resolve().parents[2]))
LLM_MODEL  = os.environ.get('PLOVER_LLM', 'gemma2:9b')
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')

ROOTCODES = [
    'AGREE','CONSULT','SUPPORT','COOPERATE','AID','YIELD',
    'REQUEST','ACCUSE','REJECT','THREATEN',
    'PROTEST','SANCTION','MOBILIZE','COERCE','ASSAULT'
]
ROOT2QUAD = {
    'AGREE':1,'CONSULT':1,'SUPPORT':1,
    'COOPERATE':2,'AID':2,'YIELD':2,
    'REQUEST':3,'ACCUSE':3,'REJECT':3,'THREATEN':3,
    'PROTEST':4,'SANCTION':4,'MOBILIZE':4,'COERCE':4,'ASSAULT':4
}
ROOT2BIN = {r: (1 if ROOT2QUAD[r] <= 2 else 2) for r in ROOTCODES}

QUAD_NAMES = {1:'Verbal Cooperation', 2:'Material Cooperation',
              3:'Verbal Conflict', 4:'Material Conflict'}

# ---------------------------------------------------------------------------
# PLOVER Knowledge Base — codebook chunks, disambiguation rules, examples
# ---------------------------------------------------------------------------

# Each chunk is a self-contained piece of codebook knowledge.
# Chunked by rootcode so retrieval can pull the 3-4 most relevant definitions.

CODEBOOK_CHUNKS = [
    # --- Verbal Cooperation (Q1) ---
    {
        "id": "cb_agree",
        "rootcode": "AGREE",
        "quadcode": 1,
        "type": "definition",
        "text": ("AGREE (Quadcode 1 - Verbal Cooperation): Agree to, offer, promise, "
                 "or otherwise indicate willingness or commitment to cooperate, including "
                 "promises to sign or ratify agreements. Cooperative actions (CONSULT, "
                 "SUPPORT, COOPERATE, AID, YIELD) reported in future tense are also coded "
                 "as AGREE. Key signal: verbal commitment to future cooperative action.")
    },
    {
        "id": "cb_consult",
        "rootcode": "CONSULT",
        "quadcode": 1,
        "type": "definition",
        "text": ("CONSULT (Quadcode 1 - Verbal Cooperation): All consultations and "
                 "meetings, including visiting and hosting visits, meeting at neutral "
                 "location, and consultation by phone or other media. Key signal: "
                 "face-to-face or mediated dialogue between parties.")
    },
    {
        "id": "cb_support",
        "rootcode": "SUPPORT",
        "quadcode": 1,
        "type": "definition",
        "text": ("SUPPORT (Quadcode 1 - Verbal Cooperation): Initiate, resume, improve, "
                 "or expand diplomatic, non-material cooperation; express support for, "
                 "commend, approve policy, action, or actor, or ratify, sign, or finalize "
                 "an agreement or treaty. Key signal: verbal endorsement or diplomatic "
                 "approval without material exchange.")
    },
    # --- Material Cooperation (Q2) ---
    {
        "id": "cb_cooperate",
        "rootcode": "COOPERATE",
        "quadcode": 2,
        "type": "definition",
        "text": ("COOPERATE (Quadcode 2 - Material Cooperation): Initiate, resume, "
                 "improve, or expand mutual material cooperation or exchange, including "
                 "economics, military, judicial matters, and sharing of intelligence. "
                 "Key signal: tangible bilateral exchange benefiting both parties.")
    },
    {
        "id": "cb_aid",
        "rootcode": "AID",
        "quadcode": 2,
        "type": "definition",
        "text": ("AID (Quadcode 2 - Material Cooperation): All provisions of providing "
                 "material aid whose material benefits primarily accrue to the recipient, "
                 "including monetary, military, humanitarian, asylum etc. "
                 "Key signal: one-directional material benefit to recipient.")
    },
    {
        "id": "cb_yield",
        "rootcode": "YIELD",
        "quadcode": 2,
        "type": "definition",
        "text": ("YIELD (Quadcode 2 - Material Cooperation): Yieldings or concessions, "
                 "such as resignations of government officials, easing of legal "
                 "restrictions, the release of prisoners, repatriation of refugees or "
                 "property, allowing third party access, disarming militarily, "
                 "implementing a ceasefire, and a military retreat. "
                 "Key signal: concession or de-escalation through material action.")
    },
    # --- Verbal Conflict (Q3) ---
    {
        "id": "cb_request",
        "rootcode": "REQUEST",
        "quadcode": 3,
        "type": "definition",
        "text": ("REQUEST (Quadcode 3 - Verbal Conflict): All verbal requests, demands, "
                 "and orders, which are less forceful than threats and potentially carry "
                 "less serious repercussions. Demands that take the form of demonstrations "
                 "or protests etc. are coded as PROTEST. "
                 "Key signal: verbal demand without threat of consequences.")
    },
    {
        "id": "cb_accuse",
        "rootcode": "ACCUSE",
        "quadcode": 3,
        "type": "definition",
        "text": ("ACCUSE (Quadcode 3 - Verbal Conflict): Express disapprovals, "
                 "objections, and complaints; condemn, decry a policy or an action; "
                 "criticize, defame, denigrate responsible parties. Accuse, allege, or "
                 "charge, both judicially and informally. Sue or bring to court. "
                 "Investigations. Key signal: verbal blame or criticism.")
    },
    {
        "id": "cb_reject",
        "rootcode": "REJECT",
        "quadcode": 3,
        "type": "definition",
        "text": ("REJECT (Quadcode 3 - Verbal Conflict): All rejections and refusals, "
                 "such as assistance, changes in policy, yielding, or meetings. "
                 "Key signal: explicit verbal refusal or denial of cooperation.")
    },
    {
        "id": "cb_threaten",
        "rootcode": "THREATEN",
        "quadcode": 3,
        "type": "definition",
        "text": ("THREATEN (Quadcode 3 - Verbal Conflict): All threats, coercive or "
                 "forceful warnings with serious potential repercussions. Threats are "
                 "generally verbal acts except for purely symbolic material actions such "
                 "as having an unarmed group place a flag on some territory. "
                 "Key signal: verbal warning of future punitive action.")
    },
    # --- Material Conflict (Q4) ---
    {
        "id": "cb_protest",
        "rootcode": "PROTEST",
        "quadcode": 4,
        "type": "definition",
        "text": ("PROTEST (Quadcode 4 - Material Conflict): All civilian demonstrations "
                 "and other collective actions carried out as protests against the "
                 "recipient: Dissent collectively, publicly show negative feelings or "
                 "opinions; rally, gather to protest a policy, action, or actor(s). "
                 "Key signal: physical collective action expressing opposition.")
    },
    {
        "id": "cb_sanction",
        "rootcode": "SANCTION",
        "quadcode": 4,
        "type": "definition",
        "text": ("SANCTION (Quadcode 4 - Material Conflict): All reductions in existing, "
                 "routine, or cooperative relations. For example, withdrawing or "
                 "discontinuing diplomatic, commercial, or material exchanges. "
                 "Key signal: withdrawal or reduction of existing cooperation.")
    },
    {
        "id": "cb_mobilize",
        "rootcode": "MOBILIZE",
        "quadcode": 4,
        "type": "definition",
        "text": ("MOBILIZE (Quadcode 4 - Material Conflict): All military or police moves "
                 "that fall short of the actual use of force. This category is different "
                 "from ASSAULT, which refers to actual uses of force, while military "
                 "posturing falls short of actual use of force and is typically a "
                 "demonstration of military capabilities and readiness. "
                 "Key signal: show of force without actual violence.")
    },
    {
        "id": "cb_coerce",
        "rootcode": "COERCE",
        "quadcode": 4,
        "type": "definition",
        "text": ("COERCE (Quadcode 4 - Material Conflict): Repression, restrictions on "
                 "rights, or coercive uses of power falling short of violence, such as "
                 "arresting, deporting, banning individuals, imposing curfew, imposing "
                 "restrictions on political freedoms or movement, conducting cyber "
                 "attacks, etc. Key signal: non-violent but coercive material action.")
    },
    {
        "id": "cb_assault",
        "rootcode": "ASSAULT",
        "quadcode": 4,
        "type": "definition",
        "text": ("ASSAULT (Quadcode 4 - Material Conflict): Deliberate actions which can "
                 "potentially result in substantial physical harm. Includes all actual "
                 "uses of military force, bombings, shootings, kidnappings, and "
                 "unconventional mass violence. "
                 "Key signal: actual physical violence or harm.")
    },
]

# Disambiguation rules from the PLOVER codebook (high-value for rootcode confusion)
DISAMBIGUATION_CHUNKS = [
    {
        "id": "rule_conflict_override",
        "type": "disambiguation",
        "text": ("CONFLICT OVERRIDE RULE: When top predictions include candidates in "
                 "both Material Conflict (Q4) and Verbal Conflict (Q3), give priority "
                 "to Material Conflict labels. Example: 'protest to request' should be "
                 "coded as PROTEST (material) not REQUEST (verbal). Similarly, 'convict "
                 "and arrest' should be COERCE (material) not ACCUSE (verbal).")
    },
    {
        "id": "rule_future_tense",
        "type": "disambiguation",
        "text": ("FUTURE TENSE RULE: Cooperative actions (CONSULT, SUPPORT, COOPERATE, "
                 "AID, YIELD) reported in future tense should be coded as AGREE. "
                 "Example: 'will provide aid' -> AGREE, not AID. "
                 "'plans to meet' -> AGREE, not CONSULT.")
    },
    {
        "id": "rule_protest_vs_request",
        "type": "disambiguation",
        "text": ("PROTEST vs REQUEST: Demands that take the form of demonstrations, "
                 "protests, or collective public action are coded as PROTEST (Q4-Material "
                 "Conflict), not REQUEST (Q3-Verbal Conflict). The physical/collective "
                 "nature overrides the verbal demand component.")
    },
    {
        "id": "rule_threaten_vs_mobilize",
        "type": "disambiguation",
        "text": ("THREATEN vs MOBILIZE: THREATEN is typically verbal — warnings with "
                 "potential repercussions. MOBILIZE is material — actual military or "
                 "police movements that demonstrate capability without using force. "
                 "If troops physically move or position, it is MOBILIZE. If a leader "
                 "issues a warning, it is THREATEN.")
    },
    {
        "id": "rule_coerce_vs_assault",
        "type": "disambiguation",
        "text": ("COERCE vs ASSAULT: COERCE covers non-violent but coercive actions "
                 "(arrests, deportations, curfews, bans, cyber attacks). ASSAULT covers "
                 "actions that result or can result in substantial physical harm "
                 "(bombings, shootings, military strikes). The threshold is physical harm.")
    },
    {
        "id": "rule_accuse_vs_reject",
        "type": "disambiguation",
        "text": ("ACCUSE vs REJECT: ACCUSE involves expressing blame, criticism, "
                 "objections, or charges. REJECT involves explicitly refusing or denying "
                 "a request, proposal, or cooperation. If the actor is blaming/criticizing "
                 "-> ACCUSE. If the actor is saying no to something -> REJECT.")
    },
    {
        "id": "rule_aid_vs_cooperate",
        "type": "disambiguation",
        "text": ("AID vs COOPERATE: AID is unidirectional — material benefits flow "
                 "primarily to the recipient (humanitarian aid, military aid, asylum). "
                 "COOPERATE is bidirectional — mutual material exchange benefiting both "
                 "parties (trade agreements, joint military exercises, intelligence sharing).")
    },
    {
        "id": "rule_support_vs_aid",
        "type": "disambiguation",
        "text": ("SUPPORT vs AID: SUPPORT is verbal/diplomatic — expressing approval, "
                 "commending, endorsing. AID is material — providing tangible resources. "
                 "If the action is words of encouragement -> SUPPORT. "
                 "If money/goods/troops are transferred -> AID.")
    },
    {
        "id": "rule_yield_vs_agree",
        "type": "disambiguation",
        "text": ("YIELD vs AGREE: YIELD involves actual material concessions (releasing "
                 "prisoners, ceasefire, retreat, easing restrictions). AGREE is verbal "
                 "commitment to future action. If concession already happened -> YIELD. "
                 "If it is a promise to concede -> AGREE.")
    },
]

# Representative labeled examples for retrieval (from PLV_test patterns)
# These serve as few-shot anchors during RAG retrieval
EXAMPLE_BANK = [
    {"text": "<S>The president</S> agreed to sign the peace treaty with <T>the opposition</T>.", "rootcode": "AGREE", "explanation": "Verbal commitment to cooperate (sign treaty)."},
    {"text": "<S>Delegates</S> held talks with <T>foreign ministers</T> at the UN summit.", "rootcode": "CONSULT", "explanation": "Meeting/consultation between parties."},
    {"text": "<S>The government</S> praised <T>the international community</T> for its efforts.", "rootcode": "SUPPORT", "explanation": "Verbal endorsement without material exchange."},
    {"text": "<S>Both countries</S> signed a mutual defense pact with <T>each other</T>.", "rootcode": "COOPERATE", "explanation": "Bilateral material cooperation (defense pact)."},
    {"text": "<S>The UN</S> delivered humanitarian supplies to <T>the refugees</T>.", "rootcode": "AID", "explanation": "Unidirectional material benefit to recipient."},
    {"text": "<S>The military</S> released political prisoners held by <T>the regime</T>.", "rootcode": "YIELD", "explanation": "Material concession (releasing prisoners)."},
    {"text": "<S>The ambassador</S> demanded <T>the foreign government</T> release the hostages.", "rootcode": "REQUEST", "explanation": "Verbal demand without threat of force."},
    {"text": "<S>Opposition leaders</S> accused <T>the ruling party</T> of corruption.", "rootcode": "ACCUSE", "explanation": "Verbal blame/criticism directed at target."},
    {"text": "<S>The parliament</S> rejected <T>the president's</T> proposed budget.", "rootcode": "REJECT", "explanation": "Explicit refusal of a proposal."},
    {"text": "<S>The general</S> warned <T>neighboring forces</T> of severe consequences.", "rootcode": "THREATEN", "explanation": "Verbal warning of punitive action."},
    {"text": "<S>Thousands of citizens</S> marched in the streets against <T>the government</T>.", "rootcode": "PROTEST", "explanation": "Collective physical action expressing opposition."},
    {"text": "<S>The EU</S> withdrew its ambassador from <T>the country</T>.", "rootcode": "SANCTION", "explanation": "Reduction/withdrawal of diplomatic relations."},
    {"text": "<S>The army</S> deployed troops along the border with <T>the neighboring state</T>.", "rootcode": "MOBILIZE", "explanation": "Military positioning without actual use of force."},
    {"text": "<S>Police</S> arrested dozens of <T>opposition activists</T>.", "rootcode": "COERCE", "explanation": "Non-violent coercive action (arrests)."},
    {"text": "<S>Rebel forces</S> launched an attack on <T>government positions</T>.", "rootcode": "ASSAULT", "explanation": "Actual use of force causing physical harm."},
    # Additional confusable-pair examples
    {"text": "<S>The president</S> will provide economic assistance to <T>the ally nation</T>.", "rootcode": "AGREE", "explanation": "Future tense cooperative action -> AGREE, not AID."},
    {"text": "<S>Protesters</S> demanded reforms from <T>the government</T> at a rally.", "rootcode": "PROTEST", "explanation": "Demand via demonstration -> PROTEST, not REQUEST (Conflict Override)."},
    {"text": "<S>Authorities</S> convicted and imprisoned <T>the dissident</T>.", "rootcode": "COERCE", "explanation": "Conviction + imprisonment -> COERCE, not ACCUSE (material overrides verbal)."},
    {"text": "<S>The navy</S> conducted exercises near <T>the disputed islands</T>.", "rootcode": "MOBILIZE", "explanation": "Military exercise = show of force, not actual violence -> MOBILIZE, not ASSAULT."},
    {"text": "<S>The foreign minister</S> expressed intent to ease tensions with <T>the rival state</T>.", "rootcode": "AGREE", "explanation": "Intent to de-escalate = verbal cooperation commitment -> AGREE."},
]

# ---------------------------------------------------------------------------
# Noise preprocessing for real-time data
# ---------------------------------------------------------------------------

def preprocess_noisy_text(text: str) -> str:
    """
    Clean noisy real-time news text for PLOVER classification.
    Handles: extra whitespace, encoding artifacts, broken HTML,
    missing S/T markers, common abbreviations, URL removal.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove HTML tags (common in scraped news)
    text = re.sub(r'<(?!/?[ST]>)[^>]+>', '', text)

    # Fix common encoding artifacts
    replacements = {
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\xa0': ' ',
        '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove leading/trailing quotes that wrap entire sentence
    if len(text) > 2 and text[0] in '""\'' and text[-1] in '""\'' :
        text = text[1:-1].strip()

    return text


def has_source_target_markers(text: str) -> bool:
    """Check if text already has <S></S> and <T></T> markers."""
    return bool(re.search(r'<S>', text) and re.search(r'<T>', text))


def add_noise_context_to_prompt(text: str) -> str:
    """
    Add a note to the prompt when source/target markers are missing,
    signaling the LLM to infer actors from context.
    """
    if has_source_target_markers(text):
        return ""
    return ("\nNOTE: This sentence lacks explicit <S>source</S> and <T>target</T> "
            "markers. Identify the primary political actors and their roles from "
            "context. The source is the actor performing the action; the target "
            "is the actor receiving it.\n")


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class PLOVERRag:
    """
    Retrieval-Augmented Generation for PLOVER classification.

    Builds a FAISS index over codebook chunks, disambiguation rules,
    and labeled examples. At inference time, retrieves the top-k most
    relevant pieces and constructs a focused prompt.
    """

    def __init__(self, embed_model_name: str = EMBED_MODEL,
                 top_k_codebook: int = 4,
                 top_k_rules: int = 2,
                 top_k_examples: int = 3,
                 use_noise_preprocessing: bool = False,
                 verbose: bool = True):
        self.top_k_codebook = top_k_codebook
        self.top_k_rules = top_k_rules
        self.top_k_examples = top_k_examples
        self.use_noise_preprocessing = use_noise_preprocessing
        self.verbose = verbose

        if verbose:
            print(f"  Loading embedding model: {embed_model_name} ...")
        self.embedder = SentenceTransformer(embed_model_name)

        # Build knowledge base
        self._build_indices()
        if verbose:
            print(f"  RAG ready: {len(self.cb_texts)} codebook chunks, "
                  f"{len(self.rule_texts)} rules, {len(self.ex_texts)} examples")

    def _build_indices(self):
        """Encode all knowledge chunks and build separate FAISS indices."""
        # Codebook definitions
        self.cb_texts = [c['text'] for c in CODEBOOK_CHUNKS]
        self.cb_meta  = CODEBOOK_CHUNKS
        cb_embs = self.embedder.encode(self.cb_texts, convert_to_numpy=True,
                                        normalize_embeddings=True)
        self.cb_index = faiss.IndexFlatIP(cb_embs.shape[1])
        self.cb_index.add(cb_embs.astype('float32'))

        # Disambiguation rules
        self.rule_texts = [r['text'] for r in DISAMBIGUATION_CHUNKS]
        self.rule_meta  = DISAMBIGUATION_CHUNKS
        rule_embs = self.embedder.encode(self.rule_texts, convert_to_numpy=True,
                                          normalize_embeddings=True)
        self.rule_index = faiss.IndexFlatIP(rule_embs.shape[1])
        self.rule_index.add(rule_embs.astype('float32'))

        # Example bank
        self.ex_texts = [e['text'] for e in EXAMPLE_BANK]
        self.ex_meta  = EXAMPLE_BANK
        ex_embs = self.embedder.encode(self.ex_texts, convert_to_numpy=True,
                                        normalize_embeddings=True)
        self.ex_index = faiss.IndexFlatIP(ex_embs.shape[1])
        self.ex_index.add(ex_embs.astype('float32'))

    def retrieve(self, sentence: str, strategy: str = 'cb'):
        """
        Retrieve relevant knowledge for a sentence.

        Strategies:
            'cb'     — codebook definitions + disambiguation rules only
            'cb_ex'  — codebook + rules + similar labeled examples
            'noisy'  — cb_ex + noise preprocessing + actor inference notes

        Returns dict with keys: codebook_chunks, rules, examples, noise_note
        """
        if self.use_noise_preprocessing or strategy == 'noisy':
            sentence = preprocess_noisy_text(sentence)

        q_emb = self.embedder.encode([sentence], convert_to_numpy=True,
                                      normalize_embeddings=True).astype('float32')

        # Always retrieve codebook and rules
        _, cb_idx = self.cb_index.search(q_emb, min(self.top_k_codebook, len(self.cb_texts)))
        _, rule_idx = self.rule_index.search(q_emb, min(self.top_k_rules, len(self.rule_texts)))

        result = {
            'codebook_chunks': [self.cb_meta[i] for i in cb_idx[0]],
            'rules': [self.rule_meta[i] for i in rule_idx[0]],
            'examples': [],
            'noise_note': '',
        }

        # Optionally retrieve examples
        if strategy in ('cb_ex', 'noisy'):
            _, ex_idx = self.ex_index.search(q_emb, min(self.top_k_examples, len(self.ex_texts)))
            result['examples'] = [self.ex_meta[i] for i in ex_idx[0]]

        # Add noise handling note if applicable
        if strategy == 'noisy':
            result['noise_note'] = add_noise_context_to_prompt(sentence)

        return result

    def build_prompt(self, sentence: str, strategy: str = 'cb') -> str:
        """
        Build a complete LLM prompt with RAG-retrieved context.

        Args:
            sentence: Input political event sentence
            strategy: 'cb', 'cb_ex', or 'noisy'

        Returns:
            Formatted prompt string ready for Ollama
        """
        if self.use_noise_preprocessing or strategy == 'noisy':
            clean_sentence = preprocess_noisy_text(sentence)
        else:
            clean_sentence = sentence

        retrieved = self.retrieve(sentence, strategy)

        # --- Build prompt sections ---
        parts = []
        parts.append("You are a political event classifier using the PLOVER ontology. "
                      "Classify the relation between source (<S></S>) and target (<T></T>).")

        # Retrieved codebook definitions
        parts.append("\nRELEVANT LABEL DEFINITIONS:")
        for i, chunk in enumerate(retrieved['codebook_chunks'], 1):
            parts.append(f"{i}. {chunk['text']}")

        # Retrieved disambiguation rules
        if retrieved['rules']:
            parts.append("\nDISAMBIGUATION RULES (apply these when choosing between similar labels):")
            for rule in retrieved['rules']:
                parts.append(f"- {rule['text']}")

        # Retrieved similar examples
        if retrieved['examples']:
            parts.append("\nSIMILAR LABELED EXAMPLES:")
            for ex in retrieved['examples']:
                parts.append(f"  Sentence: {ex['text']}")
                parts.append(f"  Label: {ex['rootcode']} — {ex['explanation']}")
                parts.append("")

        # Noise handling note
        if retrieved['noise_note']:
            parts.append(retrieved['noise_note'])

        # All valid labels for reference
        labels_str = ', '.join(ROOTCODES)
        parts.append(f"\nVALID LABELS: {labels_str}")

        # Input sentence
        parts.append(f"\nSentence: {clean_sentence}")
        parts.append("\nOutput ONLY the label name (e.g. AGREE, ASSAULT), nothing else.")

        return '\n'.join(parts)


# ---------------------------------------------------------------------------
# LLM Interface (Ollama)
# ---------------------------------------------------------------------------

def query_ollama(prompt: str, model: str = LLM_MODEL,
                 temperature: float = 0.1, max_retries: int = 3) -> str:
    """Send prompt to Ollama and return response text."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': temperature, 'num_predict': 30}
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


def extract_label(text: str) -> str:
    """Extract a PLOVER rootcode label from LLM response."""
    if not text:
        return 'UNKNOWN'
    text_upper = text.upper().strip()
    # Direct match
    for label in ROOTCODES:
        if text_upper == label or text_upper.startswith(label):
            return label
    # Search in response
    for label in ROOTCODES:
        if label in text_upper:
            return label
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_f1(y_true, y_pred):
    """Compute Binary, Quad, and Root macro F1 scores."""
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
    """Load PLV_test.tsv from the repo."""
    path = f'{REPO}/datasets/PLV_test.tsv'
    if not os.path.exists(path):
        print(f"  ERROR: Test data not found at {path}")
        print(f"  Set PLOVER_REPO env var or adjust REPO in this file.")
        sys.exit(1)
    df = pd.read_csv(path, sep='\t')
    if limit:
        df = df.head(limit)
    return df


def get_columns(df):
    """Identify sentence and label columns."""
    sent_col = 'marked_sentence' if 'marked_sentence' in df.columns else df.columns[0]
    label_col = 'gold_root' if 'gold_root' in df.columns else df.columns[-1]
    return sent_col, label_col


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def run_rag_experiment(name: str, strategy: str, save_path: str,
                       limit: int = None, noise_preprocess: bool = False):
    """
    Run a RAG-augmented LLM classification experiment on PLV_test.

    Args:
        name: Experiment name for logging
        strategy: RAG strategy ('cb', 'cb_ex', 'noisy')
        save_path: Path to save results CSV
        limit: Number of examples (None = all 1033)
        noise_preprocess: Whether to apply noise preprocessing
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Strategy: {strategy} | Model: {LLM_MODEL}")
    print(f"{'='*60}")

    # Initialize RAG
    rag = PLOVERRag(use_noise_preprocessing=noise_preprocess)

    # Load data
    df = load_test_data(limit)
    sent_col, label_col = get_columns(df)
    total = len(df)

    preds, truths, errors = [], [], 0
    start_time = time.time()

    print(f"  Running on {total} examples...\n")
    for idx, row in df.iterrows():
        sentence = str(row[sent_col])
        true_root = str(row[label_col]).upper().strip()

        # Build RAG-augmented prompt
        prompt = rag.build_prompt(sentence, strategy=strategy)

        # Query LLM
        response = query_ollama(prompt)
        pred = extract_label(response)

        if pred == 'UNKNOWN':
            errors += 1
            pred = 'REJECT'  # fallback (most common confusion target)

        preds.append(pred)
        truths.append(true_root)

        # Progress logging
        done = len(preds)
        if done % 50 == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"    [{done}/{total}] {rate:.1f} ex/s | "
                  f"ETA: {eta/60:.1f}min | Errors: {errors}")

    # Compute scores
    binary, quad, root = compute_f1(truths, preds)
    elapsed = time.time() - start_time

    print(f"\n  RESULTS ({name}):")
    print(f"    Binary F1: {binary:.1f}")
    print(f"    Quad F1:   {quad:.1f}")
    print(f"    Root F1:   {root:.1f}")
    print(f"    Errors:    {errors}/{total}")
    print(f"    Time:      {elapsed/60:.1f} min")

    # Save results
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    results_df = pd.DataFrame({
        'sentence': [str(row[sent_col]) for _, row in load_test_data(limit).iterrows()],
        'gold_root': truths,
        'pred_root': preds,
    })
    results_df.to_csv(save_path, index=False)
    print(f"    Saved to: {save_path}")

    return binary, quad, root


def print_comparison_table():
    """Print comparison table including RAG results if available."""
    print("\n" + "="*75)
    print("  PLOVER Classification — Full Comparison Table")
    print("="*75)
    print(f"  {'Method':<25} {'Binary F1':>10} {'Quad F1':>10} {'Root F1':>10}")
    print("  " + "-"*55)

    output_dir = f'{REPO}/outputs'

    # Check for RAG result files
    rag_files = {
        'RAG Codebook': 'rag_codebook.csv',
        'RAG CB + Examples': 'rag_cb_examples.csv',
        'RAG Noisy': 'rag_noisy.csv',
    }

    for name, fname in rag_files.items():
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            truths = df['gold_root'].tolist()
            preds = df['pred_root'].tolist()
            b, q, r = compute_f1(truths, preds)
            print(f"  {name:<25} {b:>10.1f} {q:>10.1f} {r:>10.1f}")
        else:
            print(f"  {name:<25} {'—':>10} {'—':>10} {'—':>10}")

    print("  " + "-"*55)
    print("  (Run --step rag_cb / rag_cb_ex / rag_noisy to populate)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global LLM_MODEL
    parser = argparse.ArgumentParser(
        description="PLOVER RAG Classification Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 supplemental/rag/plover_rag.py --step rag_cb --limit 5      # Quick test
    python3 supplemental/rag/plover_rag.py --step rag_cb                 # Full run, codebook RAG
    python3 supplemental/rag/plover_rag.py --step rag_cb_ex              # + example retrieval
    python3 supplemental/rag/plover_rag.py --step rag_noisy              # + noise preprocessing
    python3 supplemental/rag/plover_rag.py --step table                  # Show results
    python3 supplemental/rag/plover_rag.py --step demo                   # Interactive demo
        """)
    parser.add_argument('--step', required=True,
                        choices=['rag_cb', 'rag_cb_ex', 'rag_noisy', 'table', 'demo'],
                        help='Which experiment to run')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of test examples (default: all 1033)')
    parser.add_argument('--model', type=str, default=LLM_MODEL,
                        help=f'Ollama model name (default: {LLM_MODEL})')
    args = parser.parse_args()

    
    LLM_MODEL = args.model

    output_dir = f'{REPO}/outputs'
    os.makedirs(output_dir, exist_ok=True)

    if args.step == 'rag_cb':
        run_rag_experiment(
            name="RAG Codebook Only",
            strategy='cb',
            save_path=f'{output_dir}/rag_codebook.csv',
            limit=args.limit
        )

    elif args.step == 'rag_cb_ex':
        run_rag_experiment(
            name="RAG Codebook + Examples",
            strategy='cb_ex',
            save_path=f'{output_dir}/rag_cb_examples.csv',
            limit=args.limit
        )

    elif args.step == 'rag_noisy':
        run_rag_experiment(
            name="RAG Noisy (CB + Ex + Preprocessing)",
            strategy='noisy',
            save_path=f'{output_dir}/rag_noisy.csv',
            limit=args.limit,
            noise_preprocess=True
        )

    elif args.step == 'table':
        print_comparison_table()

    elif args.step == 'demo':
        print("\n  Interactive RAG Demo (type 'quit' to exit)\n")
        rag = PLOVERRag()
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