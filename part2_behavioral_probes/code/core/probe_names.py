"""CSV `behavioral_probe` labels used by the reviewer-facing package.

The current paper reports behavioral reliability for codebook-grounded
source-target event coding with five diagnostics: valid-label compliance,
definition recovery, order perturbations, generic labels, and swapped mappings.
Legacy aliases are kept below for older helper functions and cached outputs.
"""
from __future__ import annotations

# Paper diagnostics
I_LEGAL_LABELS = "Legal-label compliance"
II_DEFINITION_RECOVERY = "Definition recovery"
IVA_ORDER_FLEISS = "Order perturbation agreement (Fleiss kappa)"
IVB_ORDER_REVERSED = "Order perturbation change rate (reversed)"
IVC_ORDER_SHUFFLED = "Order perturbation change rate (shuffled)"
VI_GENERIC_ACCURACY = "Generic-label probe accuracy"
VI_GENERIC_F1 = "Generic-label probe F1"
VII_SWAPPED_ACCURACY = "Swapped-mapping probe accuracy"
VII_SWAPPED_F1 = "Swapped-mapping probe F1"

# Paper summary rows
ORIGINAL_CONDITION_ACCURACY = "Original-condition accuracy"
CODEBOOK_ALIGNMENT = "Codebook Alignment (CB-Align.)"
RULE_FOLLOWING_SCORE = "Rule-following Score (Rule-S)"

# Legacy tests retained for non-default helper functions.
IIIA_IN_CONTEXT_POSITIVE = "Legacy: positive in-context examples"
IIIB_IN_CONTEXT_NEGATIVE = "Legacy: negative in-context examples"
V_EXCLUSION_ALL = "Legacy: exclusion criteria consistency (all conditions correct)"
V_EXCLUSION_NORMAL = "Legacy: exclusion criteria (normal codebook, normal document)"
V_EXCLUSION_NORMAL_MODIFIED_DOC = "Legacy: exclusion criteria (normal codebook, modified document)"
V_EXCLUSION_MODIFIED_CODEBOOK_NORMAL_DOC = "Legacy: exclusion criteria (modified codebook, normal document)"
V_EXCLUSION_MODIFIED_CODEBOOK_MODIFIED_DOC = "Legacy: exclusion criteria (modified codebook, modified document)"

# Original-codebook predictive check used as Orig. Acc. in behavioral tables.
DEV_BASELINE_ACCURACY = ORIGINAL_CONDITION_ACCURACY
DEV_BASELINE_F1 = "Original-condition F1 (weighted)"
