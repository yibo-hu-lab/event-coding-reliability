
import os, re, json
from pathlib import Path

REPO = Path(os.environ.get("PLOVER_REPO", Path(__file__).resolve().parents[1]))
CAMEO_PDF = REPO / "codebooks" / "CAMEO.Manual.1.1b3.pdf"
OUTPUT_JSON = REPO / "plover_codebook.json"

# Section number in PDF -> CAMEO code -> PLOVER label
# PDF uses 2.1 = CAMEO 01, 2.2 = CAMEO 02, 2.3 = CAMEO 03, etc.
SECTION_MAP = {
    '2.1':  {'cameo': '01', 'plover': None,       'title': 'MAKE PUBLIC STATEMENT'},
    '2.2':  {'cameo': '02', 'plover': None,       'title': 'APPEAL'},
    '2.3':  {'cameo': '03', 'plover': 'AGREE',    'title': 'EXPRESS INTENT TO COOPERATE'},
    '2.4':  {'cameo': '04', 'plover': 'CONSULT',  'title': 'CONSULT'},
    '2.5':  {'cameo': '05', 'plover': 'SUPPORT',  'title': 'ENGAGE IN DIPLOMATIC COOPERATION'},
    '2.6':  {'cameo': '06', 'plover': 'COOPERATE','title': 'ENGAGE IN MATERIAL COOPERATION'},
    '2.7':  {'cameo': '07', 'plover': 'AID',      'title': 'PROVIDE AID'},
    '2.8':  {'cameo': '08', 'plover': 'YIELD',    'title': 'YIELD'},
    '2.9':  {'cameo': '09', 'plover': 'ACCUSE',   'title': 'INVESTIGATE'},
    '2.10': {'cameo': '10', 'plover': 'REQUEST',  'title': 'DEMAND'},
    '2.11': {'cameo': '11', 'plover': 'ACCUSE',   'title': 'DISAPPROVE'},
    '2.12': {'cameo': '12', 'plover': 'REJECT',   'title': 'REJECT'},
    '2.13': {'cameo': '13', 'plover': 'THREATEN', 'title': 'THREATEN'},
    '2.14': {'cameo': '14', 'plover': 'PROTEST',  'title': 'PROTEST'},
    '2.15': {'cameo': '15', 'plover': 'MOBILIZE', 'title': 'EXHIBIT MILITARY POSTURE'},
    '2.16': {'cameo': '16', 'plover': 'SANCTION', 'title': 'REDUCE RELATIONS'},
    '2.17': {'cameo': '17', 'plover': 'COERCE',   'title': 'COERCE'},
    '2.18': {'cameo': '18', 'plover': 'ASSAULT',  'title': 'ASSAULT'},
    '2.19': {'cameo': '19', 'plover': 'ASSAULT',  'title': 'FIGHT'},
    '2.20': {'cameo': '20', 'plover': 'ASSAULT',  'title': 'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE'},
}

PLOVER_QUAD = {
    'AGREE': 'Q1-Verbal Cooperation',
    'CONSULT': 'Q1-Verbal Cooperation',
    'SUPPORT': 'Q1-Verbal Cooperation',
    'COOPERATE': 'Q2-Material Cooperation',
    'AID': 'Q2-Material Cooperation',
    'YIELD': 'Q2-Material Cooperation',
    'REQUEST': 'Q3-Verbal Conflict',
    'ACCUSE': 'Q3-Verbal Conflict',
    'REJECT': 'Q3-Verbal Conflict',
    'THREATEN': 'Q3-Verbal Conflict',
    'PROTEST': 'Q4-Material Conflict',
    'SANCTION': 'Q4-Material Conflict',
    'MOBILIZE': 'Q4-Material Conflict',
    'COERCE': 'Q4-Material Conflict',
    'ASSAULT': 'Q4-Material Conflict',
}


def extract_text_from_pdf(pdf_path):
    """Extract text using available tools."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        print(f"  pdfplumber failed: {e}")

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        print(f"  pypdf failed: {e}")

    import subprocess
    result = subprocess.run(['pdftotext', '-layout', pdf_path, '-'],
                          capture_output=True, text=True)
    return result.stdout


def parse_sections(text):
    """
    Parse CAMEO manual by finding section headers like:
    2.3 EXPRESS INTENT TO COOPERATE
    2.10 DEMAND
    2.18 ASSAULT
    """
    sections = {}

    # Match section headers: "2.3 EXPRESS INTENT TO COOPERATE" etc.
    # The section numbers go from 2.1 to 2.20
    section_pattern = re.compile(
        r'^(2\.(?:1\d|20|[1-9]))\s+([A-Z][A-Z\s/\-&]+?)(?:\n|$)',
        re.MULTILINE
    )

    matches = list(section_pattern.finditer(text))
    print(f"  Found {len(matches)} section headers")

    for i, match in enumerate(matches):
        sec_num = match.group(1)
        title = match.group(2).strip()

        # Get text between this section and the next
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]

        # Extract the description (text before first subcode)
        cameo_code = SECTION_MAP.get(sec_num, {}).get('cameo', '')
        lines = section_text.strip().split('\n')
        description_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if description_lines:
                    break
                continue
            # Stop if we hit a subcode pattern like "031" or "0311"
            if cameo_code and re.match(rf'^{cameo_code}\d', stripped):
                break
            # Stop if we hit another section header
            if re.match(r'^2\.\d', stripped):
                break
            description_lines.append(stripped)
        description = ' '.join(description_lines)

        # Extract subcodes (e.g., 031, 0311, 032, etc.)
        subcodes = []
        if cameo_code:
            subcode_pattern = re.compile(
                rf'^({cameo_code}\d{{1,2}})\s*[-–:]?\s*(.+?)$',
                re.MULTILINE
            )
            for sm in subcode_pattern.finditer(section_text):
                sc_code = sm.group(1)
                sc_desc = sm.group(2).strip()
                if len(sc_desc) > 5:
                    subcodes.append({
                        'code': sc_code,
                        'description': sc_desc
                    })

        sections[sec_num] = {
            'section': sec_num,
            'cameo_code': cameo_code,
            'title': title,
            'description': description[:600],
            'subcodes': subcodes[:12]
        }

        print(f"  {sec_num} ({title[:40]:<40}) -> CAMEO {cameo_code} | {len(subcodes):>2} subcodes | desc: {len(description):>4} chars")

    return sections


def build_plover_json(sections):
    """Group parsed CAMEO sections by PLOVER label and build JSON."""
    plover_groups = {}

    for sec_num, info in sections.items():
        mapping = SECTION_MAP.get(sec_num)
        if not mapping or not mapping['plover']:
            continue
        plover_label = mapping['plover']
        if plover_label not in plover_groups:
            plover_groups[plover_label] = []
        plover_groups[plover_label].append(info)

    label_order = ['AGREE', 'CONSULT', 'SUPPORT', 'COOPERATE', 'AID',
                   'YIELD', 'REQUEST', 'ACCUSE', 'REJECT', 'THREATEN',
                   'PROTEST', 'SANCTION', 'MOBILIZE', 'COERCE', 'ASSAULT']

    labels = []
    for plover_label in label_order:
        groups = plover_groups.get(plover_label, [])
        cameo_codes = [g['cameo_code'] for g in groups]
        cameo_titles = [g['title'] for g in groups]

        descriptions = [g['description'] for g in groups if g['description']]
        definition = ' '.join(descriptions) if descriptions else f"Actions classified as {plover_label}."

        all_subcodes = []
        for g in groups:
            all_subcodes.extend(g['subcodes'])

        subcode_strs = [f"{sc['code']}: {sc['description']}" for sc in all_subcodes[:8]]
        clarification = "Includes: " + '; '.join(subcode_strs) if subcode_strs else ""

        entry = {
            "label": plover_label,
            "quadcode": PLOVER_QUAD[plover_label],
            "cameo_codes": cameo_codes,
            "cameo_titles": cameo_titles,
            "definition": definition[:600],
            "clarification": clarification[:600],
            "subcodes": [{"code": sc['code'], "description": sc['description']}
                        for sc in all_subcodes[:10]]
        }
        labels.append(entry)

    codebook = {
        "task_description": "Classify the political relation between the source actor (marked with <S></S>) and the target actor (marked with <T></T>). The source performs the action, the target receives or is affected by it. Choose exactly one label from the codebook below.",
        "output_reminder": "Output ONLY the label name (e.g., AGREE, ASSAULT). No explanations, no numbers, no other text.",
        "labels": labels,
        "disambiguation_rules": [
            "Material Conflict (Q4) overrides Verbal Conflict (Q3). 'protest to request' = PROTEST. 'convict and arrest' = COERCE.",
            "Future-tense cooperation = AGREE. 'agreed to provide aid' = AGREE, not AID.",
            "Halting existing cooperation = SANCTION. 'halted military aid' = SANCTION.",
            "Peacekeeping forces/workers/observers = AID, not MOBILIZE.",
            "CONSULT only when the meeting itself is the primary action reported."
        ]
    }
    return codebook


def main():
    if not os.path.exists(CAMEO_PDF):
        print(f"CAMEO PDF not found at {CAMEO_PDF}")
        return

    print(f"Reading {CAMEO_PDF}...")
    text = extract_text_from_pdf(CAMEO_PDF)
    print(f"Extracted {len(text)} characters")

    # Save raw text for debugging
    raw_path = f'{REPO}/codebooks/cameo_raw_text.txt'
    with open(raw_path, 'w') as f:
        f.write(text)
    print(f"Raw text saved to {raw_path}")

    print("\nParsing sections...")
    sections = parse_sections(text)
    print(f"\nFound {len(sections)} sections total")

    if len(sections) < 10:
        print(f"\nWARNING: Only found {len(sections)} sections (expected ~20).")
        print(f"Check {raw_path} to see the extracted text format.")
        print("Showing first 200 chars of raw text:")
        print(text[:200])
        return

    print("\nBuilding PLOVER JSON codebook...")
    codebook = build_plover_json(sections)
    print(f"Built codebook with {len(codebook['labels'])} PLOVER labels")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(codebook, f, indent=2)
    print(f"\nSaved -> {OUTPUT_JSON}")

    print(f"\n{'='*60}")
    print("CODEBOOK SUMMARY")
    print(f"{'='*60}")
    for entry in codebook['labels']:
        nsub = len(entry.get('subcodes', []))
        deflen = len(entry['definition'])
        print(f"  {entry['label']:<12} {entry['quadcode']:<25} CAMEO {str(entry['cameo_codes']):<12} {nsub:>2} subcodes  def:{deflen:>3} chars")


if __name__ == '__main__':
    main()