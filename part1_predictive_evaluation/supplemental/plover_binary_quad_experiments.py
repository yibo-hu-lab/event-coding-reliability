import os, sys, argparse, subprocess, time, re
from pathlib import Path
import pandas as pd
import requests
from sklearn.metrics import f1_score

REPO       = os.environ.get('PLOVER_REPO', str(Path(__file__).resolve().parents[1]))
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

# === ADDED: Direct label sets ===
QUADCODES = ['VERBAL_COOPERATION', 'MATERIAL_COOPERATION',
             'VERBAL_CONFLICT', 'MATERIAL_CONFLICT']
QUAD_ID   = {'VERBAL_COOPERATION':1, 'MATERIAL_COOPERATION':2,
             'VERBAL_CONFLICT':3, 'MATERIAL_CONFLICT':4}
BINCODES  = ['COOPERATION', 'CONFLICT']
BIN_ID    = {'COOPERATION':1, 'CONFLICT':2}
QUAD_ALIASES = {
    'V-COOP':'VERBAL_COOPERATION','VCOOP':'VERBAL_COOPERATION',
    'M-COOP':'MATERIAL_COOPERATION','MCOOP':'MATERIAL_COOPERATION',
    'V-CONF':'VERBAL_CONFLICT','VCONF':'VERBAL_CONFLICT',
    'M-CONF':'MATERIAL_CONFLICT','MCONF':'MATERIAL_CONFLICT',
    'VERBAL COOPERATION':'VERBAL_COOPERATION',
    'MATERIAL COOPERATION':'MATERIAL_COOPERATION',
    'VERBAL CONFLICT':'VERBAL_CONFLICT',
    'MATERIAL CONFLICT':'MATERIAL_CONFLICT',
}
BIN_ALIASES = {
    'COOP':'COOPERATION','COOPERATIVE':'COOPERATION',
    'CONF':'CONFLICT','CONFLICTUAL':'CONFLICT',
}
# === END ADDED ===

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

# === ADDED: Binary codebook ===
CODEBOOK_BINARY = """1. COOPERATION: The source actor engages in cooperative behavior toward the target. This includes verbal cooperation (agreements, consultations, diplomatic support, promises) AND material cooperation (aid, trade, concessions, yielding, ceasefire, release of prisoners).

2. CONFLICT: The source actor engages in conflictual behavior toward the target. This includes verbal conflict (demands, accusations, rejections, threats) AND material conflict (protests, sanctions, military mobilization, coercion, arrests, assaults, violence)."""

# === ADDED: Quadcode codebook ===
CODEBOOK_QUADCODE = """1. VERBAL_COOPERATION: The source cooperates with the target through verbal or diplomatic means — agreements, consultations, meetings, diplomatic support, promises, approvals, or signing treaties.

2. MATERIAL_COOPERATION: The source cooperates with the target through material or physical means — providing aid (monetary, military, humanitarian), trade, exchanges, concessions, yielding, ceasefire, release of prisoners, allowing access, or disarming.

3. VERBAL_CONFLICT: The source engages in verbal conflict with the target — demands, requests, accusations, criticisms, rejections, refusals, or threats. Expressed through words and statements, without physical actions.

4. MATERIAL_CONFLICT: The source engages in material/physical conflict with the target — protests, demonstrations, sanctions, military mobilization, coercion (arrests, bans, deportations, cyber attacks), or assault (violence causing physical harm).

RULE: Prioritize Material Conflict over Verbal Conflict. If an event involves both, classify as MATERIAL_CONFLICT."""
# === END ADDED ===

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

# === ADDED: Binary ICL examples ===
ICL_EXAMPLES_BINARY = """Example 1:
Sentence: <S>French National Assembly president</S> held talks with leaders of <T>Romania's</T> new government.
Answer: COOPERATION

Example 2:
Sentence: <S>The US</S> provided $500M in humanitarian aid to <T>Yemen</T>.
Answer: COOPERATION

Example 3:
Sentence: <S>A Brazilian federal court</S> rejected a request from <T>former President Lula</T>.
Answer: CONFLICT

Example 4:
Sentence: <S>Afghan rebels</S> kidnapped 16 <T>Soviet civilian advisers</T> and exploded bombs.
Answer: CONFLICT

Example 5:
Sentence: <S>A Taliban leader</S> surrendered in <T>Afghanistan's Faryab province</T>.
Answer: COOPERATION

Example 6:
Sentence: <S>North Korea</S> issued a warning of serious consequences to <T>South Korea</T>.
Answer: CONFLICT

Example 7:
Sentence: <S>Thousands of workers</S> staged demonstrations against <T>the government</T>.
Answer: CONFLICT

Example 8:
Sentence: <S>The EU</S> signed a trade agreement with <T>Canada</T>.
Answer: COOPERATION"""

# === ADDED: Quadcode ICL examples ===
ICL_EXAMPLES_QUADCODE = """Example 1:
Sentence: <S>French National Assembly president</S> held talks with leaders of <T>Romania's</T> new government.
Answer: VERBAL_COOPERATION

Example 2:
Sentence: <S>The US</S> provided $500M in humanitarian aid to <T>Yemen</T>.
Answer: MATERIAL_COOPERATION

Example 3:
Sentence: <S>A Brazilian federal court</S> rejected a request from <T>former President Lula</T>.
Answer: VERBAL_CONFLICT

Example 4:
Sentence: <S>Afghan rebels</S> kidnapped 16 <T>Soviet civilian advisers</T> and exploded bombs.
Answer: MATERIAL_CONFLICT

Example 5:
Sentence: <S>A Taliban leader</S> surrendered in <T>Afghanistan's Faryab province</T>.
Answer: MATERIAL_COOPERATION

Example 6:
Sentence: <S>North Korea</S> issued a warning of serious consequences to <T>South Korea</T>.
Answer: VERBAL_CONFLICT

Example 7:
Sentence: <S>Thousands of workers</S> staged demonstrations against <T>the government</T>.
Answer: MATERIAL_CONFLICT

Example 8:
Sentence: <S>The EU</S> signed a trade agreement with <T>Canada</T>.
Answer: VERBAL_COOPERATION"""
# === END ADDED ===


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def query_ollama(prompt, retries=3):
    for i in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                'model': LLM_MODEL, 'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.0, 'num_predict': 60}
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


# === ADDED: Binary & Quadcode extractors ===
def extract_binary(text):
    text_up = text.upper().strip()
    for label in BINCODES:
        if text_up == label: return label
    for alias, canonical in BIN_ALIASES.items():
        if alias in text_up: return canonical
    for label in BINCODES:
        if label in text_up: return label
    return 'UNKNOWN'

def extract_binary_cot(text):
    text_clean = re.sub(r'</?label>', '', text)
    m = re.search(r'ANSWER:\s*(\w+)', text.upper())
    if m:
        c = m.group(1)
        if c in BIN_ID: return c
        if c in BIN_ALIASES: return BIN_ALIASES[c]
    return extract_binary(text)

def extract_quadcode(text):
    text_up = text.upper().strip().replace('-','_').replace(' ','_')
    for label in QUADCODES:
        if text_up == label: return label
    for alias, canonical in sorted(QUAD_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias.replace('-','_').replace(' ','_') in text_up: return canonical
    for label in QUADCODES:
        if label in text_up: return label
    if 'MATERIAL' in text_up and 'COOP' in text_up: return 'MATERIAL_COOPERATION'
    if 'VERBAL' in text_up and 'COOP' in text_up: return 'VERBAL_COOPERATION'
    if 'MATERIAL' in text_up and 'CONF' in text_up: return 'MATERIAL_CONFLICT'
    if 'VERBAL' in text_up and 'CONF' in text_up: return 'VERBAL_CONFLICT'
    return 'UNKNOWN'

def extract_quadcode_cot(text):
    text_clean = re.sub(r'</?label>', '', text)
    m = re.search(r'ANSWER:\s*([\w_\- ]+)', text.upper())
    if m:
        result = extract_quadcode(m.group(1).strip())
        if result != 'UNKNOWN': return result
    return extract_quadcode(text)
# === END ADDED ===


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


# === ADDED: Direct-level F1 ===
def compute_f1_binary_direct(yt, yp):
    return f1_score(yt, yp, average='macro', labels=[1,2], zero_division=0) * 100

def compute_f1_quad_direct(yt, yp):
    return f1_score(yt, yp, average='macro', labels=[1,2,3,4], zero_division=0) * 100
# === END ADDED ===


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


# === ADDED: Direct Binary experiment runner ===
def run_llm_binary_experiment(name, prompt_fn, save_path, extractor=extract_binary, limit=None):
    df = load_test_data(limit)
    sent_col, label_col = get_columns(df)
    preds, truths, errors = [], [], 0

    print(f"\n  Running {name} on {len(df)} examples...")
    for idx, row in df.iterrows():
        sentence = str(row[sent_col])
        true_root = str(row[label_col]).upper().strip()
        true_bin = 'COOPERATION' if ROOT2BIN.get(true_root,0)==1 else 'CONFLICT'
        truths.append(true_bin)
        response = query_ollama(prompt_fn(sentence))
        pred = extractor(response)
        if pred == 'UNKNOWN':
            errors += 1
            pred = 'CONFLICT'
        preds.append(pred)
        if (idx + 1) % 100 == 0:
            yt = [BIN_ID.get(t,0) for t in truths]
            yp = [BIN_ID.get(p,0) for p in preds]
            print(f"  [{idx+1}/{len(df)}] BinF1={compute_f1_binary_direct(yt,yp):.1f} Unk={errors}")

    pd.DataFrame({
        'sentence': df[sent_col].values[:len(preds)],
        'gold_binary': truths, 'pred_binary': preds
    }).to_csv(save_path, index=False)

    yt = [BIN_ID.get(t,0) for t in truths]
    yp = [BIN_ID.get(p,0) for p in preds]
    f = compute_f1_binary_direct(yt, yp)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Binary F1 (direct) = {f:.1f}")
    print(f"  Unknown={errors}/{len(df)}  Saved={save_path}")
    print(f"{'='*60}")
    return f

# === ADDED: Direct Quadcode experiment runner ===
def run_llm_quad_experiment(name, prompt_fn, save_path, extractor=extract_quadcode, limit=None):
    df = load_test_data(limit)
    sent_col, label_col = get_columns(df)
    preds, truths, errors = [], [], 0
    qid2name = {1:'VERBAL_COOPERATION',2:'MATERIAL_COOPERATION',
                3:'VERBAL_CONFLICT',4:'MATERIAL_CONFLICT'}

    print(f"\n  Running {name} on {len(df)} examples...")
    for idx, row in df.iterrows():
        sentence = str(row[sent_col])
        true_root = str(row[label_col]).upper().strip()
        true_quad = qid2name.get(ROOT2QUAD.get(true_root,0),'UNKNOWN')
        truths.append(true_quad)
        response = query_ollama(prompt_fn(sentence))
        pred = extractor(response)
        if pred == 'UNKNOWN':
            errors += 1
            pred = 'VERBAL_CONFLICT'
        preds.append(pred)
        if (idx + 1) % 100 == 0:
            yt = [QUAD_ID.get(t,0) for t in truths]
            yp = [QUAD_ID.get(p,0) for p in preds]
            print(f"  [{idx+1}/{len(df)}] QuadF1={compute_f1_quad_direct(yt,yp):.1f} Unk={errors}")

    pd.DataFrame({
        'sentence': df[sent_col].values[:len(preds)],
        'gold_quad': truths, 'pred_quad': preds
    }).to_csv(save_path, index=False)

    yt = [QUAD_ID.get(t,0) for t in truths]
    yp = [QUAD_ID.get(p,0) for p in preds]
    f = compute_f1_quad_direct(yt, yp)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Quad F1 (direct) = {f:.1f}")
    print(f"  Unknown={errors}/{len(df)}  Saved={save_path}")
    print(f"{'='*60}")
    return f
# === END ADDED ===


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


# === ADDED: Direct Binary step runners ===
def run_llm_bin_no_cb(limit=None):
    def prompt_fn(sentence):
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>) as either COOPERATION or CONFLICT.\n\n"
                f"Sentence: {sentence}\n\n"
                f"Output ONLY one label: COOPERATION or CONFLICT")
    return run_llm_binary_experiment("LLM Bin No CB", prompt_fn,
                              f'{REPO}/outputs/llm_bin_no_cb.csv', limit=limit)

def run_llm_bin_compact(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier. Classify the relation "
                f"between source (<S></S>) and target (<T></T>).\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK_BINARY}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Output ONLY one label: COOPERATION or CONFLICT")
    return run_llm_binary_experiment("LLM Bin Compact", prompt_fn,
                              f'{REPO}/outputs/llm_bin_compact.csv', limit=limit)

def run_llm_bin_cot(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier.\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK_BINARY}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Think step by step:\n"
                f"1. Who is source, who is target?\n"
                f"2. What is the main action?\n"
                f"3. Is it cooperative or conflictual?\n\n"
                f"After reasoning, write your final answer on a new line in this exact format:\nANSWER: <label>\nwhere <label> is either COOPERATION or CONFLICT.")
    return run_llm_binary_experiment("LLM Bin CoT", prompt_fn,
                              f'{REPO}/outputs/llm_bin_cot.csv',
                              extractor=extract_binary_cot, limit=limit)

def run_llm_bin_icl(limit=None):
    def prompt_fn(sentence):
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>) as either COOPERATION or CONFLICT.\n\n"
                f"Here are labeled examples:\n\n{ICL_EXAMPLES_BINARY}\n\n"
                f"Now classify:\nSentence: {sentence}\n\n"
                f"Output ONLY one label: COOPERATION or CONFLICT")
    return run_llm_binary_experiment("LLM Bin ICL", prompt_fn,
                              f'{REPO}/outputs/llm_bin_icl.csv', limit=limit)

# === ADDED: Direct Quadcode step runners ===
def run_llm_quad_no_cb(limit=None):
    def prompt_fn(sentence):
        labels = ', '.join(QUADCODES)
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>).\n\n"
                f"Sentence: {sentence}\n\n"
                f"Choose one label from: {labels}\n\n"
                f"Output ONLY the label name, nothing else.")
    return run_llm_quad_experiment("LLM Quad No CB", prompt_fn,
                              f'{REPO}/outputs/llm_quad_no_cb.csv', limit=limit)

def run_llm_quad_compact(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier. Classify the relation "
                f"between source (<S></S>) and target (<T></T>).\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK_QUADCODE}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Output ONLY one label: VERBAL_COOPERATION, MATERIAL_COOPERATION, "
                f"VERBAL_CONFLICT, or MATERIAL_CONFLICT")
    return run_llm_quad_experiment("LLM Quad Compact", prompt_fn,
                              f'{REPO}/outputs/llm_quad_compact.csv', limit=limit)

def run_llm_quad_cot(limit=None):
    def prompt_fn(sentence):
        return (f"You are a political event classifier.\n\n"
                f"LABEL DEFINITIONS:\n{CODEBOOK_QUADCODE}\n\n"
                f"Sentence: {sentence}\n\n"
                f"Think step by step:\n"
                f"1. Who is source, who is target?\n"
                f"2. What is the main action?\n"
                f"3. Verbal or material?\n"
                f"4. Cooperative or conflictual?\n\n"
                f"After reasoning, write final answer as:\n"
                f"ANSWER: <label>\nwhere <label> is one of: VERBAL_COOPERATION, MATERIAL_COOPERATION, "
                f"VERBAL_CONFLICT, MATERIAL_CONFLICT")
    return run_llm_quad_experiment("LLM Quad CoT", prompt_fn,
                              f'{REPO}/outputs/llm_quad_cot.csv',
                              extractor=extract_quadcode_cot, limit=limit)

def run_llm_quad_icl(limit=None):
    def prompt_fn(sentence):
        labels = ', '.join(QUADCODES)
        return (f"Classify the political relation between source (<S></S>) "
                f"and target (<T></T>).\n\n"
                f"Here are labeled examples:\n\n{ICL_EXAMPLES_QUADCODE}\n\n"
                f"Now classify:\nSentence: {sentence}\n\n"
                f"Labels: {labels}\nOutput ONLY the label name, nothing else.")
    return run_llm_quad_experiment("LLM Quad ICL", prompt_fn,
                              f'{REPO}/outputs/llm_quad_icl.csv', limit=limit)
# === END ADDED ===


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
                            ('LLM ICL','llm_icl.csv')]:
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

    # === ADDED: Direct Binary results ===
    print(f"\n{'--- Direct Binary ---':<28} {'Binary':>8}")
    print('─' * 38)
    for name, csv_name in [('LLM Bin No CB','llm_bin_no_cb.csv'),
                            ('LLM Bin Compact','llm_bin_compact.csv'),
                            ('LLM Bin CoT','llm_bin_cot.csv'),
                            ('LLM Bin ICL','llm_bin_icl.csv')]:
        path = f'{REPO}/outputs/{csv_name}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            yt = [BIN_ID.get(t,0) for t in df['gold_binary'].str.upper().tolist()]
            yp = [BIN_ID.get(p,0) for p in df['pred_binary'].str.upper().tolist()]
            f = compute_f1_binary_direct(yt, yp)
            print(f"{name:<28} {f:>8.1f}")
            rows.append({'Method':name,'Binary_F1':round(f,1),'Quad_F1':'—','Root_F1':'—','Avg':'—'})
        else:
            print(f"{name:<28} {'—':>8}")

    # === ADDED: Direct Quadcode results ===
    print(f"\n{'--- Direct Quadcode ---':<28} {'Quad':>8}")
    print('─' * 38)
    for name, csv_name in [('LLM Quad No CB','llm_quad_no_cb.csv'),
                            ('LLM Quad Compact','llm_quad_compact.csv'),
                            ('LLM Quad CoT','llm_quad_cot.csv'),
                            ('LLM Quad ICL','llm_quad_icl.csv')]:
        path = f'{REPO}/outputs/{csv_name}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            yt = [QUAD_ID.get(t,0) for t in df['gold_quad'].str.upper().tolist()]
            yp = [QUAD_ID.get(p,0) for p in df['pred_quad'].str.upper().tolist()]
            f = compute_f1_quad_direct(yt, yp)
            print(f"{name:<28} {f:>8.1f}")
            rows.append({'Method':name,'Binary_F1':'—','Quad_F1':round(f,1),'Root_F1':'—','Avg':'—'})
        else:
            print(f"{name:<28} {'—':>8}")
    # === END ADDED ===

    out = f'{REPO}/outputs/final_results.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved → {out}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='PLOVER Experiments')
    parser.add_argument('--step', required=True,
        choices=['tree','tiny','full','llm_no_cb','llm_compact','llm_cot','llm_icl',
                 # === ADDED ===
                 'llm_bin_no_cb','llm_bin_compact','llm_bin_cot','llm_bin_icl',
                 'llm_quad_no_cb','llm_quad_compact','llm_quad_cot','llm_quad_icl',
                 'llm_bin_all','llm_quad_all',
                 # === END ADDED ===
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
        'llm_cot':   lambda: run_llm_cot(args.limit),
        'llm_icl':   lambda: run_llm_icl(args.limit),
        'nli_all':   lambda: [run_zsp_tree(), run_zsp_tiny(), run_zsp_full()],
        'llm_all':   lambda: [run_llm_no_codebook(args.limit),
                              run_llm_compact(args.limit),
                              run_llm_cot(args.limit),
                              run_llm_icl(args.limit)],
        'table':     lambda: print_final_table(),
        'all':       lambda: [run_zsp_tree(), run_zsp_tiny(), run_zsp_full(),
                              run_llm_no_codebook(args.limit),
                              run_llm_compact(args.limit),
                              run_llm_cot(args.limit),
                              run_llm_icl(args.limit),
                              # === ADDED ===
                              run_llm_bin_no_cb(args.limit),
                              run_llm_bin_compact(args.limit),
                              run_llm_bin_cot(args.limit),
                              run_llm_bin_icl(args.limit),
                              run_llm_quad_no_cb(args.limit),
                              run_llm_quad_compact(args.limit),
                              run_llm_quad_cot(args.limit),
                              run_llm_quad_icl(args.limit),
                              # === END ADDED ===
                              print_final_table()],
        # === ADDED ===
        'llm_bin_no_cb':  lambda: run_llm_bin_no_cb(args.limit),
        'llm_bin_compact': lambda: run_llm_bin_compact(args.limit),
        'llm_bin_cot':    lambda: run_llm_bin_cot(args.limit),
        'llm_bin_icl':    lambda: run_llm_bin_icl(args.limit),
        'llm_bin_all':    lambda: [run_llm_bin_no_cb(args.limit),
                                   run_llm_bin_compact(args.limit),
                                   run_llm_bin_cot(args.limit),
                                   run_llm_bin_icl(args.limit)],
        'llm_quad_no_cb': lambda: run_llm_quad_no_cb(args.limit),
        'llm_quad_compact': lambda: run_llm_quad_compact(args.limit),
        'llm_quad_cot':   lambda: run_llm_quad_cot(args.limit),
        'llm_quad_icl':   lambda: run_llm_quad_icl(args.limit),
        'llm_quad_all':   lambda: [run_llm_quad_no_cb(args.limit),
                                   run_llm_quad_compact(args.limit),
                                   run_llm_quad_cot(args.limit),
                                   run_llm_quad_icl(args.limit)],
        # === END ADDED ===
    }
    dispatch[args.step]()

    if args.step != 'table':
        print(f"\n  → Run 'python3 supplemental/plover_binary_quad_experiments.py --step table' for full comparison.")

if __name__ == '__main__':
    main()