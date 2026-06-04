#!/usr/bin/env python3
"""
plover_rag_v2_experiments.py — Enhanced RAG module for PLOVER political event classification

Key improvements over v1:
  1. Hierarchical retrieval: classify quadcode first, then retrieve only
     rootcodes within that quadcode (mirrors ZSP tree L1→L2→L3 funnel)
  2. Contrastive example pairs for confusable rootcodes
  3. Forced confusable-neighbor inclusion in retrieved context
  4. Upgraded embedding model option (all-mpnet-base-v2)
  5. Reordered prompt: sentence first, then evidence
  6. Expanded example bank (~40 examples including contrastive pairs)

Reference Enriched prompt result: Gemma2:9b → Binary=95.8, Quad=86.8, Root=71.4

Usage:
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_hier           # Hierarchical RAG (main)
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_hier_ex         # Hierarchical + examples
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_cb              # Flat RAG (v1 baseline)
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_cb_ex           # Flat + examples
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_noisy           # Noise-robust variant
    python3 supplemental/rag/plover_rag_v2_experiments.py --step table               # Comparison table
    python3 supplemental/rag/plover_rag_v2_experiments.py --step demo                # Interactive demo
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_hier --limit 5  # Quick test
"""

import os, sys, re, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# ML dependency check
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install sentence-transformers faiss-cpu")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO       = os.environ.get('PLOVER_REPO', str(Path(__file__).resolve().parents[2]))
LLM_MODEL  = os.environ.get('PLOVER_LLM', 'gemma2:9b')
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')

# === CHANGE 3: Upgraded default embedding model ===
# all-mpnet-base-v2 gives better semantic discrimination than MiniLM
# Set EMBED_MODEL=all-MiniLM-L6-v2 if memory is tight
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-mpnet-base-v2')

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

QUAD_NAMES = {
    1: 'Verbal Cooperation',  2: 'Material Cooperation',
    3: 'Verbal Conflict',     4: 'Material Conflict'
}
QUAD_ROOTCODES = {
    1: ['AGREE', 'CONSULT', 'SUPPORT'],
    2: ['COOPERATE', 'AID', 'YIELD'],
    3: ['REQUEST', 'ACCUSE', 'REJECT', 'THREATEN'],
    4: ['PROTEST', 'SANCTION', 'MOBILIZE', 'COERCE', 'ASSAULT'],
}

# === CHANGE 4: Confusable neighbor map ===
# When we retrieve a rootcode definition, always also include its
# most-confused neighbor(s). Derived from the ZSP confusion matrix
# in Hu et al. Table 15 and Enriched-codebook error patterns.
CONFUSABLE_NEIGHBORS = {
    'AGREE':     ['SUPPORT', 'CONSULT'],
    'CONSULT':   ['AGREE', 'SUPPORT'],
    'SUPPORT':   ['AGREE', 'COOPERATE'],
    'COOPERATE': ['SUPPORT', 'AID'],
    'AID':       ['COOPERATE', 'YIELD'],
    'YIELD':     ['AID', 'AGREE'],
    'REQUEST':   ['ACCUSE', 'PROTEST'],
    'ACCUSE':    ['REQUEST', 'REJECT'],
    'REJECT':    ['ACCUSE', 'THREATEN'],
    'THREATEN':  ['REJECT', 'MOBILIZE'],
    'PROTEST':   ['REQUEST', 'MOBILIZE'],
    'SANCTION':  ['COERCE', 'REJECT'],
    'MOBILIZE':  ['THREATEN', 'PROTEST'],
    'COERCE':    ['ASSAULT', 'SANCTION'],
    'ASSAULT':   ['COERCE', 'MOBILIZE'],
}

# ---------------------------------------------------------------------------
# PLOVER Knowledge Base — codebook chunks
# ---------------------------------------------------------------------------

CODEBOOK_CHUNKS = [
    # --- Verbal Cooperation (Q1) ---
    {"id": "cb_agree", "rootcode": "AGREE", "quadcode": 1, "type": "definition",
     "text": ("AGREE (Quadcode 1 - Verbal Cooperation): Agree to, offer, promise, "
              "or otherwise indicate willingness or commitment to cooperate, including "
              "promises to sign or ratify agreements. Cooperative actions (CONSULT, "
              "SUPPORT, COOPERATE, AID, YIELD) reported in future tense are also coded "
              "as AGREE. Key signal: verbal commitment to future cooperative action.")},
    {"id": "cb_consult", "rootcode": "CONSULT", "quadcode": 1, "type": "definition",
     "text": ("CONSULT (Quadcode 1 - Verbal Cooperation): All consultations and "
              "meetings, including visiting and hosting visits, meeting at neutral "
              "location, and consultation by phone or other media. Key signal: "
              "face-to-face or mediated dialogue between parties.")},
    {"id": "cb_support", "rootcode": "SUPPORT", "quadcode": 1, "type": "definition",
     "text": ("SUPPORT (Quadcode 1 - Verbal Cooperation): Initiate, resume, improve, "
              "or expand diplomatic, non-material cooperation; express support for, "
              "commend, approve policy, action, or actor, or ratify, sign, or finalize "
              "an agreement or treaty. Key signal: verbal endorsement or diplomatic "
              "approval without material exchange.")},
    # --- Material Cooperation (Q2) ---
    {"id": "cb_cooperate", "rootcode": "COOPERATE", "quadcode": 2, "type": "definition",
     "text": ("COOPERATE (Quadcode 2 - Material Cooperation): Initiate, resume, "
              "improve, or expand mutual material cooperation or exchange, including "
              "economics, military, judicial matters, and sharing of intelligence. "
              "Key signal: tangible bilateral exchange benefiting both parties.")},
    {"id": "cb_aid", "rootcode": "AID", "quadcode": 2, "type": "definition",
     "text": ("AID (Quadcode 2 - Material Cooperation): All provisions of providing "
              "material aid whose material benefits primarily accrue to the recipient, "
              "including monetary, military, humanitarian, asylum etc. "
              "Key signal: one-directional material benefit to recipient.")},
    {"id": "cb_yield", "rootcode": "YIELD", "quadcode": 2, "type": "definition",
     "text": ("YIELD (Quadcode 2 - Material Cooperation): All concessions, retreats, "
              "releases, allowances of rights, easing of restrictions, ceasefires, "
              "reduction of protests, and policy reforms by actors who previously "
              "opposed them. Key signal: concession from a previously opposing position.")},
    # --- Verbal Conflict (Q3) ---
    {"id": "cb_request", "rootcode": "REQUEST", "quadcode": 3, "type": "definition",
     "text": ("REQUEST (Quadcode 3 - Verbal Conflict): Requests, appeals, and demands "
              "for something that do not carry an explicit threat of force. Also "
              "includes sending people to investigate. Key signal: verbal demand "
              "without coercive implication.")},
    {"id": "cb_accuse", "rootcode": "ACCUSE", "quadcode": 3, "type": "definition",
     "text": ("ACCUSE (Quadcode 3 - Verbal Conflict): Accusations, blame, complaints, "
              "lawsuits, or public verbal attacks directed at a target. Key signal: "
              "verbal blame, criticism, or legal action against another party.")},
    {"id": "cb_reject", "rootcode": "REJECT", "quadcode": 3, "type": "definition",
     "text": ("REJECT (Quadcode 3 - Verbal Conflict): Explicit refusals of proposals, "
              "demands, or agreements. Rejecting cooperation, consultation, yielding, "
              "or any other action. Key signal: refusal or defiance of a specific "
              "request or proposal.")},
    {"id": "cb_threaten", "rootcode": "THREATEN", "quadcode": 3, "type": "definition",
     "text": ("THREATEN (Quadcode 3 - Verbal Conflict): Warnings of punitive action, "
              "ultimatums, and threats of military or economic consequences. Also "
              "includes cooperative actions reported in FUTURE tense for Conflict "
              "quadcodes (since future conflict intent = threat). Key signal: verbal "
              "warning of negative consequences.")},
    # --- Material Conflict (Q4) ---
    {"id": "cb_protest", "rootcode": "PROTEST", "quadcode": 4, "type": "definition",
     "text": ("PROTEST (Quadcode 4 - Material Conflict): Demonstrations, rallies, "
              "strikes, boycotts, obstruction of activities, and other forms of "
              "collective non-violent physical action opposing a target. Key signal: "
              "collective physical action expressing opposition.")},
    {"id": "cb_sanction", "rootcode": "SANCTION", "quadcode": 4, "type": "definition",
     "text": ("SANCTION (Quadcode 4 - Material Conflict): Withdrawal or reduction of "
              "diplomatic relations, economic sanctions, halting negotiations, "
              "expelling people or organizations. Key signal: withdrawal of prior "
              "cooperation or formal punitive reduction in relations.")},
    {"id": "cb_mobilize", "rootcode": "MOBILIZE", "quadcode": 4, "type": "definition",
     "text": ("MOBILIZE (Quadcode 4 - Material Conflict): Increasing military forces, "
              "heightened alert, preparation of forces, military exercises, or "
              "positioning troops — without actually using force. Key signal: "
              "military positioning or readiness without violence.")},
    {"id": "cb_coerce", "rootcode": "COERCE", "quadcode": 4, "type": "definition",
     "text": ("COERCE (Quadcode 4 - Material Conflict): Repression, restrictions on "
              "rights, or coercive uses of power falling short of violence, such as "
              "arresting, deporting, banning individuals, imposing curfew, imposing "
              "restrictions on political freedoms or movement, conducting cyber "
              "attacks, etc. Key signal: non-violent but coercive material action.")},
    {"id": "cb_assault", "rootcode": "ASSAULT", "quadcode": 4, "type": "definition",
     "text": ("ASSAULT (Quadcode 4 - Material Conflict): Deliberate actions which can "
              "potentially result in substantial physical harm. Includes all actual "
              "uses of military force, bombings, shootings, kidnappings, and "
              "unconventional mass violence. "
              "Key signal: actual physical violence or harm.")},
]

# Quick lookup: rootcode -> chunk
CB_BY_ROOT = {c['rootcode']: c for c in CODEBOOK_CHUNKS}

# ---------------------------------------------------------------------------
# Disambiguation rules
# ---------------------------------------------------------------------------

DISAMBIGUATION_CHUNKS = [
    {"id": "rule_conflict_override", "type": "disambiguation",
     "text": ("CONFLICT OVERRIDE RULE: When top predictions include candidates in "
              "both Material Conflict (Q4) and Verbal Conflict (Q3), give priority "
              "to Material Conflict labels. Example: 'protest to request' should be "
              "coded as PROTEST (material) not REQUEST (verbal). Similarly, 'convict "
              "and arrest' should be COERCE (material) not ACCUSE (verbal).")},
    {"id": "rule_future_tense", "type": "disambiguation",
     "text": ("FUTURE TENSE RULE: Cooperative actions (CONSULT, SUPPORT, COOPERATE, "
              "AID, YIELD) reported in future tense should be coded as AGREE. "
              "Example: 'will provide aid' -> AGREE, not AID. "
              "'plans to meet' -> AGREE, not CONSULT.")},
    {"id": "rule_protest_vs_request", "type": "disambiguation",
     "text": ("PROTEST vs REQUEST: Demands that take the form of demonstrations, "
              "protests, or collective public action are coded as PROTEST (Q4), "
              "not REQUEST (Q3). Physical/collective nature overrides verbal demand.")},
    {"id": "rule_threaten_vs_mobilize", "type": "disambiguation",
     "text": ("THREATEN vs MOBILIZE: THREATEN is verbal — warnings with potential "
              "repercussions. MOBILIZE is material — actual military/police movements "
              "demonstrating capability without using force. Troops physically move "
              "-> MOBILIZE. Leader issues warning -> THREATEN.")},
    {"id": "rule_accuse_vs_reject", "type": "disambiguation",
     "text": ("ACCUSE vs REJECT: ACCUSE is blame or criticism directed at a target. "
              "REJECT is refusal of a specific proposal/demand. If the source is "
              "denying a request -> REJECT. If criticizing behavior -> ACCUSE.")},
    {"id": "rule_coerce_vs_assault", "type": "disambiguation",
     "text": ("COERCE vs ASSAULT: COERCE is non-violent coercive action (arrests, "
              "deportation, bans, curfews, cyber attacks). ASSAULT involves actual "
              "physical violence or potential for substantial physical harm. "
              "Arrests -> COERCE. Bombings/shootings -> ASSAULT.")},
    {"id": "rule_support_vs_aid", "type": "disambiguation",
     "text": ("SUPPORT vs AID: SUPPORT is verbal/diplomatic (endorse, approve, sign). "
              "AID involves material transfer (money, supplies, asylum). If something "
              "tangible changes hands -> AID. If it is an endorsement -> SUPPORT.")},
    {"id": "rule_yield_vs_agree", "type": "disambiguation",
     "text": ("YIELD vs AGREE: YIELD requires a prior opposing position that is now "
              "conceded. AGREE is verbal commitment to future cooperation. "
              "If concession already happened -> YIELD. Promise to concede -> AGREE.")},
    {"id": "rule_cooperate_vs_aid", "type": "disambiguation",
     "text": ("COOPERATE vs AID: COOPERATE is mutual/bilateral exchange benefiting "
              "both sides (joint exercises, intelligence sharing). AID is unidirectional "
              "benefit flowing mainly to the recipient. Both sides gain -> COOPERATE. "
              "One side receives -> AID.")},
]

# ---------------------------------------------------------------------------
# === CHANGE 2: Expanded example bank with contrastive pairs ===
# ~40 examples including explicit contrastive pairs for confusable rootcodes
# ---------------------------------------------------------------------------

EXAMPLE_BANK = [
    # --- Core examples (one per rootcode) ---
    {"text": "<S>The president</S> agreed to sign the peace treaty with <T>the opposition</T>.",
     "rootcode": "AGREE", "explanation": "Verbal commitment to cooperate (sign treaty)."},
    {"text": "<S>Delegates</S> held talks with <T>foreign ministers</T> at the UN summit.",
     "rootcode": "CONSULT", "explanation": "Meeting/consultation between parties."},
    {"text": "<S>The government</S> praised <T>the international community</T> for its efforts.",
     "rootcode": "SUPPORT", "explanation": "Verbal endorsement without material exchange."},
    {"text": "<S>Both countries</S> signed a mutual defense pact with <T>each other</T>.",
     "rootcode": "COOPERATE", "explanation": "Bilateral material cooperation (defense pact)."},
    {"text": "<S>The UN</S> delivered humanitarian supplies to <T>the refugees</T>.",
     "rootcode": "AID", "explanation": "Unidirectional material benefit to recipient."},
    {"text": "<S>The military</S> released political prisoners held by <T>the regime</T>.",
     "rootcode": "YIELD", "explanation": "Material concession (releasing prisoners)."},
    {"text": "<S>The ambassador</S> demanded <T>the foreign government</T> release the hostages.",
     "rootcode": "REQUEST", "explanation": "Verbal demand without threat of force."},
    {"text": "<S>Opposition leaders</S> accused <T>the ruling party</T> of corruption.",
     "rootcode": "ACCUSE", "explanation": "Verbal blame/criticism directed at target."},
    {"text": "<S>The parliament</S> rejected <T>the president's</T> proposed budget.",
     "rootcode": "REJECT", "explanation": "Explicit refusal of a proposal."},
    {"text": "<S>The general</S> warned <T>neighboring forces</T> of severe consequences.",
     "rootcode": "THREATEN", "explanation": "Verbal warning of punitive action."},
    {"text": "<S>Thousands of citizens</S> marched in the streets against <T>the government</T>.",
     "rootcode": "PROTEST", "explanation": "Collective physical action expressing opposition."},
    {"text": "<S>The EU</S> withdrew its ambassador from <T>the country</T>.",
     "rootcode": "SANCTION", "explanation": "Reduction/withdrawal of diplomatic relations."},
    {"text": "<S>The army</S> deployed troops along the border with <T>the neighboring state</T>.",
     "rootcode": "MOBILIZE", "explanation": "Military positioning without actual use of force."},
    {"text": "<S>Police</S> arrested dozens of <T>opposition activists</T>.",
     "rootcode": "COERCE", "explanation": "Non-violent coercive action (arrests)."},
    {"text": "<S>Rebel forces</S> launched an attack on <T>government positions</T>.",
     "rootcode": "ASSAULT", "explanation": "Actual use of force causing physical harm."},

    # --- CONTRASTIVE PAIRS: future tense (AGREE vs actual action) ---
    {"text": "<S>The president</S> will provide economic assistance to <T>the ally nation</T>.",
     "rootcode": "AGREE", "explanation": "CONTRASTIVE: Future tense cooperative -> AGREE, not AID."},
    {"text": "<S>The president</S> provided economic assistance to <T>the ally nation</T>.",
     "rootcode": "AID", "explanation": "CONTRASTIVE: Past tense material transfer -> AID, not AGREE."},
    {"text": "<S>The foreign minister</S> plans to meet with <T>the delegation</T> next week.",
     "rootcode": "AGREE", "explanation": "CONTRASTIVE: Future meeting -> AGREE, not CONSULT."},
    {"text": "<S>The foreign minister</S> met with <T>the delegation</T> yesterday.",
     "rootcode": "CONSULT", "explanation": "CONTRASTIVE: Completed meeting -> CONSULT, not AGREE."},
    {"text": "<S>The government</S> promised to ease restrictions on <T>the opposition</T>.",
     "rootcode": "AGREE", "explanation": "CONTRASTIVE: Promise to yield -> AGREE, not YIELD."},
    {"text": "<S>The government</S> eased restrictions on <T>the opposition</T>.",
     "rootcode": "YIELD", "explanation": "CONTRASTIVE: Actual concession -> YIELD, not AGREE."},

    # --- CONTRASTIVE PAIRS: verbal vs material conflict ---
    {"text": "<S>Protesters</S> demanded reforms from <T>the government</T> at a rally.",
     "rootcode": "PROTEST", "explanation": "CONTRASTIVE: Demand via demonstration -> PROTEST (Q4), not REQUEST (Q3)."},
    {"text": "<S>The ambassador</S> called on <T>the government</T> to implement reforms.",
     "rootcode": "REQUEST", "explanation": "CONTRASTIVE: Verbal appeal without physical action -> REQUEST."},
    {"text": "<S>Authorities</S> convicted and imprisoned <T>the dissident</T>.",
     "rootcode": "COERCE", "explanation": "CONTRASTIVE: Material action (imprisonment) -> COERCE, not ACCUSE."},
    {"text": "<S>Authorities</S> publicly blamed <T>the dissident</T> for inciting unrest.",
     "rootcode": "ACCUSE", "explanation": "CONTRASTIVE: Verbal blame only -> ACCUSE, not COERCE."},

    # --- CONTRASTIVE PAIRS: coerce vs assault ---
    {"text": "<S>Security forces</S> detained <T>journalists</T> covering the protests.",
     "rootcode": "COERCE", "explanation": "CONTRASTIVE: Detention without violence -> COERCE, not ASSAULT."},
    {"text": "<S>Security forces</S> fired on <T>protesters</T> with live ammunition.",
     "rootcode": "ASSAULT", "explanation": "CONTRASTIVE: Physical violence -> ASSAULT, not COERCE."},

    # --- CONTRASTIVE PAIRS: threaten vs mobilize ---
    {"text": "<S>The defense minister</S> warned <T>the enemy</T> of military retaliation.",
     "rootcode": "THREATEN", "explanation": "CONTRASTIVE: Verbal warning -> THREATEN, not MOBILIZE."},
    {"text": "<S>The navy</S> conducted exercises near <T>the disputed islands</T>.",
     "rootcode": "MOBILIZE", "explanation": "CONTRASTIVE: Physical military positioning -> MOBILIZE, not THREATEN."},

    # --- CONTRASTIVE PAIRS: support vs cooperate ---
    {"text": "<S>The president</S> endorsed <T>the ally's</T> position at the summit.",
     "rootcode": "SUPPORT", "explanation": "CONTRASTIVE: Verbal endorsement -> SUPPORT, not COOPERATE."},
    {"text": "<S>The two nations</S> launched a joint intelligence-sharing program with <T>each other</T>.",
     "rootcode": "COOPERATE", "explanation": "CONTRASTIVE: Mutual material exchange -> COOPERATE, not SUPPORT."},

    # --- CONTRASTIVE PAIRS: cooperate vs aid ---
    {"text": "<S>Both countries</S> exchanged military technology with <T>each other</T>.",
     "rootcode": "COOPERATE", "explanation": "CONTRASTIVE: Bilateral exchange -> COOPERATE, not AID."},
    {"text": "<S>The donor country</S> shipped medical supplies to <T>the disaster zone</T>.",
     "rootcode": "AID", "explanation": "CONTRASTIVE: One-way material transfer -> AID, not COOPERATE."},

    # --- CONTRASTIVE PAIRS: accuse vs reject ---
    {"text": "<S>The spokesperson</S> criticized <T>the foreign government's</T> human rights record.",
     "rootcode": "ACCUSE", "explanation": "CONTRASTIVE: Verbal criticism -> ACCUSE, not REJECT."},
    {"text": "<S>The committee</S> voted against <T>the proposed resolution</T>.",
     "rootcode": "REJECT", "explanation": "CONTRASTIVE: Refusing a specific proposal -> REJECT, not ACCUSE."},

    # --- CONTRASTIVE PAIRS: sanction vs reject ---
    {"text": "<S>The country</S> imposed trade restrictions on <T>the rival state</T>.",
     "rootcode": "SANCTION", "explanation": "CONTRASTIVE: Material punitive action -> SANCTION, not REJECT."},
    {"text": "<S>The country</S> refused to participate in <T>the trade talks</T>.",
     "rootcode": "REJECT", "explanation": "CONTRASTIVE: Verbal refusal -> REJECT, not SANCTION."},

    # --- Additional edge cases from Hu et al. confusion patterns ---
    {"text": "<S>The premier</S> expressed intent to ease tensions with <T>the rival state</T>.",
     "rootcode": "AGREE", "explanation": "Intent to de-escalate = verbal cooperation commitment -> AGREE."},
    {"text": "<S>The government</S> signed the ceasefire agreement with <T>rebel forces</T>.",
     "rootcode": "SUPPORT", "explanation": "Signing/ratifying an agreement -> SUPPORT (diplomatic action), not AGREE."},
    {"text": "<S>The regime</S> shut down newspapers operated by <T>the opposition</T>.",
     "rootcode": "COERCE", "explanation": "Restricting press freedom -> COERCE (non-violent material restriction)."},
    {"text": "<S>Armed groups</S> kidnapped <T>aid workers</T> in the region.",
     "rootcode": "ASSAULT", "explanation": "Kidnapping = potential for substantial physical harm -> ASSAULT."},
]

# ---------------------------------------------------------------------------
# Noise preprocessing (unchanged from v1)
# ---------------------------------------------------------------------------

def preprocess_noisy_text(text: str) -> str:
    """Clean noisy real-time news text."""
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


def has_source_target_markers(text: str) -> bool:
    """Check if text has <S></S> and <T></T> markers."""
    return bool(re.search(r'<S>.*?</S>', text) and re.search(r'<T>.*?</T>', text))


def add_noise_context_to_prompt(text: str) -> str:
    """Add noise-handling instructions if markers are missing."""
    if not has_source_target_markers(text):
        return ("\nNOTE: This sentence may be missing <S>/<T> actor markers. "
                "Identify the most likely source (initiator of the action) and "
                "target (receiver) from context, then classify the relation.")
    return ""


# ---------------------------------------------------------------------------
# PLOVERRag v2 — main class
# ---------------------------------------------------------------------------

class PLOVERRag:
    """Enhanced RAG engine with hierarchical retrieval."""

    def __init__(self, embed_model: str = EMBED_MODEL,
                 top_k_codebook: int = 4, top_k_rules: int = 3,
                 top_k_examples: int = 4,
                 use_noise_preprocessing: bool = False):
        print(f"  Loading embedding model: {embed_model} ...")
        self.embedder = SentenceTransformer(embed_model)
        self.top_k_codebook = top_k_codebook
        self.top_k_rules = top_k_rules
        self.top_k_examples = top_k_examples
        self.use_noise_preprocessing = use_noise_preprocessing

        # Build FAISS indices
        print("  Building FAISS indices ...")

        # Codebook definitions index
        self.cb_texts = [c['text'] for c in CODEBOOK_CHUNKS]
        self.cb_meta  = CODEBOOK_CHUNKS
        cb_embs = self.embedder.encode(self.cb_texts, convert_to_numpy=True,
                                        normalize_embeddings=True)
        self.cb_index = faiss.IndexFlatIP(cb_embs.shape[1])
        self.cb_index.add(cb_embs.astype('float32'))

        # Disambiguation rules index
        self.rule_texts = [r['text'] for r in DISAMBIGUATION_CHUNKS]
        self.rule_meta  = DISAMBIGUATION_CHUNKS
        rule_embs = self.embedder.encode(self.rule_texts, convert_to_numpy=True,
                                          normalize_embeddings=True)
        self.rule_index = faiss.IndexFlatIP(rule_embs.shape[1])
        self.rule_index.add(rule_embs.astype('float32'))

        # Example bank index
        self.ex_texts = [e['text'] for e in EXAMPLE_BANK]
        self.ex_meta  = EXAMPLE_BANK
        ex_embs = self.embedder.encode(self.ex_texts, convert_to_numpy=True,
                                        normalize_embeddings=True)
        self.ex_index = faiss.IndexFlatIP(ex_embs.shape[1])
        self.ex_index.add(ex_embs.astype('float32'))

        print("  RAG engine ready.\n")

    # -----------------------------------------------------------------------
    # Retrieval methods
    # -----------------------------------------------------------------------

    def retrieve_flat(self, sentence: str, include_examples: bool = False):
        """Original flat retrieval across all rootcodes (v1 behavior)."""
        if self.use_noise_preprocessing:
            sentence = preprocess_noisy_text(sentence)

        q_emb = self.embedder.encode([sentence], convert_to_numpy=True,
                                      normalize_embeddings=True).astype('float32')

        _, cb_idx = self.cb_index.search(q_emb, min(self.top_k_codebook, len(self.cb_texts)))
        _, rule_idx = self.rule_index.search(q_emb, min(self.top_k_rules, len(self.rule_texts)))

        # === CHANGE 4: Force-include confusable neighbors ===
        retrieved_roots = set()
        codebook_chunks = []
        for i in cb_idx[0]:
            chunk = self.cb_meta[i]
            codebook_chunks.append(chunk)
            retrieved_roots.add(chunk['rootcode'])

        # Add neighbors for each retrieved rootcode
        neighbors_to_add = set()
        for root in retrieved_roots:
            for neighbor in CONFUSABLE_NEIGHBORS.get(root, []):
                if neighbor not in retrieved_roots:
                    neighbors_to_add.add(neighbor)

        for neighbor in neighbors_to_add:
            if neighbor in CB_BY_ROOT:
                codebook_chunks.append(CB_BY_ROOT[neighbor])

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

    def retrieve_hierarchical(self, sentence: str, include_examples: bool = False):
        """
        === CHANGE 1: Two-stage hierarchical retrieval ===

        Stage 1: Predict quadcode using a lightweight prompt with only the
                 4 quadcode descriptions. This leverages your existing 86.8%
                 quadcode accuracy.
        Stage 2: Retrieve only rootcode definitions within the predicted
                 quadcode, plus confusable neighbors that might cross the
                 quadcode boundary.
        """
        if self.use_noise_preprocessing:
            sentence = preprocess_noisy_text(sentence)

        # Stage 1: Predict quadcode
        predicted_quad = self._predict_quadcode(sentence)

        # Stage 2: Get rootcodes for this quadcode
        target_roots = set(QUAD_ROOTCODES.get(predicted_quad, ROOTCODES))

        # Also add cross-boundary neighbors for edge cases
        for root in list(target_roots):
            for neighbor in CONFUSABLE_NEIGHBORS.get(root, []):
                if ROOT2QUAD[neighbor] != predicted_quad:
                    # Only add the closest cross-boundary neighbor, not all
                    target_roots.add(neighbor)

        # Collect codebook chunks for target rootcodes
        codebook_chunks = [CB_BY_ROOT[r] for r in target_roots if r in CB_BY_ROOT]

        # Retrieve disambiguation rules (still use semantic search)
        q_emb = self.embedder.encode([sentence], convert_to_numpy=True,
                                      normalize_embeddings=True).astype('float32')
        _, rule_idx = self.rule_index.search(q_emb, min(self.top_k_rules, len(self.rule_texts)))

        result = {
            'codebook_chunks': codebook_chunks,
            'rules': [self.rule_meta[i] for i in rule_idx[0]],
            'examples': [],
            'noise_note': '',
            'predicted_quad': predicted_quad,
        }

        if include_examples:
            # Retrieve examples, but prefer those from the predicted quadcode
            _, ex_idx = self.ex_index.search(q_emb, min(self.top_k_examples * 2, len(self.ex_texts)))
            candidates = [self.ex_meta[i] for i in ex_idx[0]]
            # Prioritize same-quadcode examples
            same_quad = [e for e in candidates if ROOT2QUAD.get(e['rootcode']) == predicted_quad]
            other = [e for e in candidates if ROOT2QUAD.get(e['rootcode']) != predicted_quad]
            result['examples'] = (same_quad + other)[:self.top_k_examples]

        return result

    def _predict_quadcode(self, sentence: str) -> int:
        """Lightweight quadcode classification for the hierarchical pipeline."""
        quad_prompt = (
            "Classify this political event into one of four categories.\n\n"
            "Categories:\n"
            "1. VERBAL_COOPERATION: Verbal agreements, consultations, meetings, "
            "diplomatic support, endorsements, promises to cooperate\n"
            "2. MATERIAL_COOPERATION: Material aid, bilateral exchange, concessions, "
            "releasing prisoners, providing supplies, mutual cooperation\n"
            "3. VERBAL_CONFLICT: Accusations, demands, requests, threats, refusals, "
            "verbal warnings, rejections of proposals\n"
            "4. MATERIAL_CONFLICT: Protests, sanctions, military mobilization, "
            "arrests, violence, bombings, kidnappings, coercion\n\n"
            f"Sentence: {sentence}\n\n"
            "Output ONLY the number (1, 2, 3, or 4)."
        )
        response = query_ollama(quad_prompt, temperature=0.0)
        # Extract the number
        for char in response.strip():
            if char in '1234':
                return int(char)
        # Fallback: use semantic similarity to quadcode descriptions
        return self._fallback_quadcode(sentence)

    def _fallback_quadcode(self, sentence: str) -> int:
        """Fallback quadcode prediction via embedding similarity."""
        quad_descriptions = [
            "verbal cooperation agreement meeting consultation support endorsement",
            "material cooperation aid supply exchange concession release",
            "verbal conflict accusation demand request threat rejection warning",
            "material conflict protest sanction mobilize arrest violence attack coerce"
        ]
        q_emb = self.embedder.encode([sentence], normalize_embeddings=True)
        q_embs = self.embedder.encode(quad_descriptions, normalize_embeddings=True)
        sims = np.dot(q_emb, q_embs.T)[0]
        return int(np.argmax(sims)) + 1  # 1-indexed

    # -----------------------------------------------------------------------
    # === CHANGE 5: Prompt building with sentence-first ordering ===
    # -----------------------------------------------------------------------

    def build_prompt(self, sentence: str, strategy: str = 'hier') -> str:
        """
        Build LLM prompt with RAG-retrieved context.

        Strategies:
            'cb'       — flat retrieval, codebook + rules only
            'cb_ex'    — flat retrieval + examples
            'hier'     — hierarchical (quadcode-first) + rules
            'hier_ex'  — hierarchical + examples
            'noisy'    — hier_ex + noise preprocessing
        """
        clean = preprocess_noisy_text(sentence) if (
            self.use_noise_preprocessing or strategy == 'noisy') else sentence

        # Select retrieval method
        if strategy in ('hier', 'hier_ex', 'noisy'):
            include_ex = strategy in ('hier_ex', 'noisy')
            retrieved = self.retrieve_hierarchical(clean, include_examples=include_ex)
        else:
            include_ex = strategy == 'cb_ex'
            retrieved = self.retrieve_flat(clean, include_examples=include_ex)

        parts = []

        # === CHANGE 5: Sentence FIRST, then evidence ===
        parts.append("You are a political event classifier using the PLOVER ontology.")
        parts.append(f"\nSENTENCE TO CLASSIFY:\n{clean}")

        # If hierarchical, tell the model which quadcode was pre-selected
        if strategy in ('hier', 'hier_ex', 'noisy') and 'predicted_quad' in retrieved:
            quad_name = QUAD_NAMES.get(retrieved['predicted_quad'], '?')
            parts.append(f"\nPRE-CLASSIFICATION: This event is likely {quad_name}.")
            quad_roots = QUAD_ROOTCODES.get(retrieved['predicted_quad'], ROOTCODES)
            parts.append(f"Primary candidates: {', '.join(quad_roots)}")

        # Retrieved codebook definitions
        parts.append("\nRELEVANT LABEL DEFINITIONS:")
        seen_roots = set()
        for i, chunk in enumerate(retrieved['codebook_chunks'], 1):
            if chunk['rootcode'] not in seen_roots:
                parts.append(f"{i}. {chunk['text']}")
                seen_roots.add(chunk['rootcode'])

        # Retrieved disambiguation rules
        if retrieved['rules']:
            parts.append("\nDISAMBIGUATION RULES:")
            for rule in retrieved['rules']:
                parts.append(f"- {rule['text']}")

        # Retrieved examples (contrastive pairs will naturally cluster)
        if retrieved['examples']:
            parts.append("\nSIMILAR LABELED EXAMPLES:")
            for ex in retrieved['examples']:
                parts.append(f"  Input: {ex['text']}")
                parts.append(f"  Label: {ex['rootcode']} ({ex['explanation']})")
                parts.append("")

        # Noise handling
        if strategy == 'noisy':
            noise_note = add_noise_context_to_prompt(clean)
            if noise_note:
                parts.append(noise_note)

        # Valid labels
        labels_str = ', '.join(ROOTCODES)
        parts.append(f"\nVALID LABELS: {labels_str}")
        parts.append("\nOutput ONLY the label name (e.g. AGREE, ASSAULT), nothing else.")

        return '\n'.join(parts)


# ---------------------------------------------------------------------------
# LLM Interface (Ollama)
# ---------------------------------------------------------------------------

def query_ollama(prompt: str, model: str = None,
                 temperature: float = 0.1, max_retries: int = 3) -> str:
    """Send prompt to Ollama and return response text."""
    if model is None:
        model = LLM_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 30}
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


def extract_label(response: str) -> str:
    """Extract a valid PLOVER rootcode from LLM response."""
    if not response:
        return "UNKNOWN"
    text = response.upper().strip()
    # Try exact match first
    for root in ROOTCODES:
        if text == root:
            return root
    # Try to find rootcode anywhere in response
    for root in ROOTCODES:
        if root in text:
            return root
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_f1(truths, preds):
    """Compute macro F1 at Binary, Quadcode, and Rootcode levels."""
    def safe_f1(y_true, y_pred):
        labels = sorted(set(y_true) | set(y_pred))
        return f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0) * 100

    truth_bin  = [ROOT2BIN.get(r, 0) for r in truths]
    pred_bin   = [ROOT2BIN.get(r, 0) for r in preds]
    truth_quad = [ROOT2QUAD.get(r, 0) for r in truths]
    pred_quad  = [ROOT2QUAD.get(r, 0) for r in preds]

    return safe_f1(truth_bin, pred_bin), safe_f1(truth_quad, pred_quad), safe_f1(truths, preds)


def load_test_data(limit=None):
    """Load PLV_test.tsv."""
    path = f'{REPO}/datasets/PLV_test.tsv'
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        sys.exit(1)
    df = pd.read_csv(path, sep='\t')
    if limit:
        df = df.head(limit)
    return df


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    'rag_cb':       'cb',
    'rag_cb_ex':    'cb_ex',
    'rag_hier':     'hier',
    'rag_hier_ex':  'hier_ex',
    'rag_noisy':    'noisy',
}


def run_rag_experiment(name: str, strategy: str, save_path: str,
                       limit=None, noise_preprocess=False):
    """Run a full RAG experiment on PLV_test.tsv."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Strategy: {strategy} | Model: {LLM_MODEL}")
    print(f"  Embed model: {EMBED_MODEL}")
    print(f"{'='*70}\n")

    rag = PLOVERRag(use_noise_preprocessing=noise_preprocess)
    df = load_test_data(limit)

    # Detect sentence column
    sent_col = 'sentence' if 'sentence' in df.columns else df.columns[0]
    root_col = 'rootcode' if 'rootcode' in df.columns else 'gold_root'
    if root_col not in df.columns:
        # Try to find the rootcode column
        for c in df.columns:
            if 'root' in c.lower() or 'label' in c.lower():
                root_col = c
                break

    truths = []
    preds  = []
    total  = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        sentence = str(row[sent_col])
        gold     = str(row[root_col]).strip().upper()

        prompt   = rag.build_prompt(sentence, strategy=strategy)
        response = query_ollama(prompt)
        pred     = extract_label(response)

        truths.append(gold)
        preds.append(pred)

        if (idx + 1) % 50 == 0 or idx == 0:
            b, q, r = compute_f1(truths, preds)
            print(f"  [{idx+1:>4}/{total}]  Binary={b:.1f}  Quad={q:.1f}  Root={r:.1f}  "
                  f"last_pred={pred}  gold={gold}")

    # Final scores
    binary, quad, root = compute_f1(truths, preds)
    print(f"\n  FINAL — Binary={binary:.1f}  Quad={quad:.1f}  Root={root:.1f}")
    print(f"  Reference: Binary=95.8  Quad=86.8  Root=71.4  (Gemma2:9b Enriched)")

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df = pd.DataFrame({
        'sentence': [str(row[sent_col]) for _, row in load_test_data(limit).iterrows()],
        'gold_root': truths,
        'pred_root': preds,
    })
    results_df.to_csv(save_path, index=False)
    print(f"  Saved to: {save_path}")

    return binary, quad, root


def print_comparison_table():
    """Print comparison table including all RAG variants."""
    print(f"\n{'='*80}")
    print("  PLOVER Classification — RAG v2 Comparison Table")
    print(f"{'='*80}")
    print(f"  {'Method':<30} {'Binary F1':>10} {'Quad F1':>10} {'Root F1':>10}")
    print("  " + "-"*60)

    # Known baselines
    baselines = [
        ('Gemma2:9b Enriched',         95.8, 86.8, 71.4),
        ('Prior ZSP Tree',             96.4, 89.6, 82.4),
    ]
    for name, b, q, r in baselines:
        print(f"  {name:<30} {b:>10.1f} {q:>10.1f} {r:>10.1f}")

    print("  " + "-"*60)

    output_dir = f'{REPO}/outputs'
    rag_files = {
        'RAG v2 Flat (CB)':         'rag_v2_cb.csv',
        'RAG v2 Flat (CB+Ex)':      'rag_v2_cb_ex.csv',
        'RAG v2 Hierarchical':      'rag_v2_hier.csv',
        'RAG v2 Hierarchical+Ex':   'rag_v2_hier_ex.csv',
        'RAG v2 Noisy':             'rag_v2_noisy.csv',
        # v1 files for comparison
        'RAG v1 Codebook':          'rag_codebook.csv',
        'RAG v1 CB+Examples':       'rag_cb_examples.csv',
        'RAG v1 Noisy':             'rag_noisy.csv',
    }

    for name, fname in rag_files.items():
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            truths = df['gold_root'].tolist()
            preds  = df['pred_root'].tolist()
            b, q, r = compute_f1(truths, preds)
            print(f"  {name:<30} {b:>10.1f} {q:>10.1f} {r:>10.1f}")
        else:
            print(f"  {name:<30} {'—':>10} {'—':>10} {'—':>10}")

    print("  " + "-"*60)
    print("  (Run --step rag_hier / rag_hier_ex / rag_cb / rag_cb_ex / rag_noisy)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global LLM_MODEL, EMBED_MODEL
    parser = argparse.ArgumentParser(
        description="PLOVER RAG v2 — Enhanced Classification Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_hier --limit 5    # Quick test
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_hier               # Full hierarchical run
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_hier_ex             # + contrastive examples
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_cb                  # Flat RAG (v1 style)
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_cb_ex               # Flat + examples
    python3 supplemental/rag/plover_rag_v2_experiments.py --step rag_noisy               # Noise-robust
    python3 supplemental/rag/plover_rag_v2_experiments.py --step table                   # Comparison table
    python3 supplemental/rag/plover_rag_v2_experiments.py --step demo                    # Interactive demo
        """)
    parser.add_argument('--step', required=True,
                        choices=['rag_cb', 'rag_cb_ex', 'rag_hier', 'rag_hier_ex',
                                 'rag_noisy', 'table', 'demo'],
                        help='Which experiment to run')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of test examples (default: all 1033)')
    parser.add_argument('--model', type=str, default=LLM_MODEL,
                        help=f'Ollama model name (default: {LLM_MODEL})')
    parser.add_argument('--embed', type=str, default=EMBED_MODEL,
                        help=f'Embedding model (default: {EMBED_MODEL})')
    args = parser.parse_args()

    
    LLM_MODEL = args.model
    EMBED_MODEL = args.embed

    output_dir = f'{REPO}/outputs'
    os.makedirs(output_dir, exist_ok=True)

    strategy = STRATEGY_MAP.get(args.step, args.step)

    if args.step == 'table':
        print_comparison_table()
    elif args.step == 'demo':
        print("\n  Interactive RAG v2 Demo (type 'quit' to exit)\n")
        rag = PLOVERRag(embed_model=EMBED_MODEL)
        while True:
            sentence = input("\n  Enter sentence: ").strip()
            if sentence.lower() in ('quit', 'exit', 'q'):
                break
            for s in ['hier', 'hier_ex', 'cb']:
                prompt = rag.build_prompt(sentence, strategy=s)
                response = query_ollama(prompt)
                pred = extract_label(response)
                print(f"  [{s:>8}] -> {pred}  (prompt: {len(prompt)} chars)")
    else:
        step_to_filename = {
            'rag_cb':      'rag_v2_cb.csv',
            'rag_cb_ex':   'rag_v2_cb_ex.csv',
            'rag_hier':    'rag_v2_hier.csv',
            'rag_hier_ex': 'rag_v2_hier_ex.csv',
            'rag_noisy':   'rag_v2_noisy.csv',
        }
        fname = step_to_filename[args.step]
        noise = args.step == 'rag_noisy'

        run_rag_experiment(
            name=f"RAG v2 — {args.step}",
            strategy=strategy,
            save_path=f'{output_dir}/{fname}',
            limit=args.limit,
            noise_preprocess=noise
        )

    print("\nDone.")


if __name__ == '__main__':
    main()