import re
import os
import random
from pathlib import Path
from typing import List, Optional, Union
from tqdm.auto import tqdm

import jinja2.exceptions
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

from core.paths import CODEBOOK_DIR


def _attn_implementation() -> str:
    """Default sdpa; flash_attention_2 needs flash-attn."""
    v = os.environ.get("DATAVERSE_ATTN_IMPLEMENTATION", "").strip()
    if v:
        return v
    return "sdpa"


def _hf_from_pretrained_kw() -> dict:
    """Map DATAVERSE_LOCAL_FILES_ONLY=1 to local_files_only=True."""
    if os.environ.get("DATAVERSE_LOCAL_FILES_ONLY", "").strip().lower() in ("1", "true", "yes"):
        return {"local_files_only": True}
    return {}


def load_model(model_name: str, 
               quantization: str = "4"):
    """Load causal LM; 4/8-bit via BitsAndBytes; dtype from GPU capability."""
    if not torch.backends.cuda.is_built():
        raise RuntimeError(
            "PyTorch has no CUDA (CPU-only build). In Colab, switch the runtime to GPU, "
            "run the dependency cell in run_pipeline.ipynb, restart the session, and run that "
            "cell again, then check: import torch; assert torch.backends.cuda.is_built()."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU visible (torch.cuda.is_available() is false). Use a GPU runtime and reconnect."
        )
    if torch.cuda.get_device_capability(0)[0] < 8:
        print("This GPU does not support bfloat16. Using float16 instead.")
        compute_type = torch.float16
    else:
        print("Using bfloat16 for model computation.")
        compute_type = torch.bfloat16

    if quantization == "4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_type,
        )
    elif quantization == "8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=compute_type,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=_attn_implementation(),
            torch_dtype=compute_type,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            **_hf_from_pretrained_kw(),
        )
        model.config.use_cache = False
        return model

    attn = _attn_implementation()
    _hf_kw = _hf_from_pretrained_kw()
    if "Phi" in model_name:
        # Phi: avoid flash-attn; eager is safest if sdpa misbehaves
        phi_attn = "eager" if attn == "flash_attention_2" else attn
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=phi_attn,
            device_map="auto",
            low_cpu_mem_usage=True,
            **_hf_kw,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation=attn,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            **_hf_kw,
        )
    model.config.use_cache = False
    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, **_hf_from_pretrained_kw()
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generate requires left padding (HF warns on right-padding).
    tokenizer.padding_side = "left"
    return tokenizer


def pad_token_id_for_generate(tokenizer) -> int:
    """Qwen etc. often have unk_token_id=None; model.generate() needs a concrete pad_token_id."""
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    uid = getattr(tokenizer, "unk_token_id", None)
    if uid is not None:
        return int(uid)
    return 0



def parse_manifestos(fn: str, 
                     party_info_df: pd.DataFrame):
    """
    Parse the manifestos into a DataFrame with the following columns:
    - party
    - text
    - meta
    - context

    Parameters
    ----------
    fn : str
        The filename of the manifesto to parse.
    party_info_df : DataFrame
        A DataFrame containing the party information, download from the Manifesto Project.
    
    Returns
    -------
    df : DataFrame
        A DataFrame containing the parsed manifesto, with one row per quasi-sentence.
    """
    # get the filename itself, not the dirs
    fn_last = os.path.basename(fn)
    party_id, _ = fn_last.split("_")
    party_info = party_info_df[party_info_df['party'] == int(party_id)].to_dict("records")[0]
    df = pd.read_csv(fn)
    if 'text' not in df.columns:
        df['text'] = df['content']
        del df['content']
    # The "text" field contains a sentence or "quasi-sentence" from the manifesto.
    # We need to go through each and add the previous sentence to a new "context"
    # field. If the previous text is a sentence fragment, we need to keep going back
    # until we find a period.
    df = df.reset_index(drop=True)
    df['meta'] = f"an excerpt from a political party in {party_info['countryname']}."

    df['context'] = ""
    for n, row in df.iterrows():
        if n == 0:
            df.at[n, 'context'] = ""
            continue
        elif n == 1:
            previous_text = df.at[n-1, 'text']
        else:
            # get two previous texts
            previous_text = df.at[n-2, 'text'] + " " + df.at[n-1, 'text']
        df.at[n, 'context'] = previous_text
    return df



def make_prompt_llama3(system_message, user_message, answer=""):
    """
    DEPRECATED: use the transformer tokenizer instead.
    Llama 3.x instruct prompt shape matches Meta Llama 3 docs: header tags + `<|eot_id|>` between turns.
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{answer}"
    )


def make_prompt_mistral_llama2(system_message, user_message, answer=""):
    prompt = f"""[INST]
<<SYS>>
{system_message}

<</SYS>>
{user_message}

---------
[/INST]
Label: {answer}"""
    return prompt


def make_prompt_phi3(system_message, user_message, answer=""):
    prompt = f"""<|user|>
{system_message}

----------------

{user_message}

----------------
<|assistant|>
Label: {answer}"""
    return prompt


def make_prompt_openai(system_message, user_message, answer=""):
    prompt = [{"role": "system", "content": system_message},
              {"role": "user", "content": user_message}]
    return prompt


def default_codebook_paths(dataset: str) -> tuple[Path, Path]:
    """Default structured codebook path plus a legacy raw-text fallback path."""
    d = dataset.lower()
    if d == "ccc":
        path = CODEBOOK_DIR / "ccc_codebook_new_format.txt"
        return path, path
    if d == "bfrs":
        path = CODEBOOK_DIR / "bfrs_codebook_new_format.txt"
        return path, path
    if d == "manifestos":
        path = CODEBOOK_DIR / "manifesto_codebook_new_hand.txt"
        return path, path
    if d == "plover":
        path = CODEBOOK_DIR / "plover_enriched_codebook.txt"
        return path, path
    if d == "aw":
        path = CODEBOOK_DIR / "aw_enriched_codebook.txt"
        return path, path
    raise ValueError("Invalid dataset. Must be one of 'ccc', 'bfrs', 'manifestos', 'plover', or 'aw'")


def resolve_codebook_paths(dataset: str) -> tuple[Path, Path]:
    """
    Resolve codebook file paths for a dataset slug.

    Slug is only a label unless you rely on built-in defaults (bfrs/ccc/manifestos/plover).
    Otherwise set any path via environment variables (`BEHAVIOR_*`, or legacy `DATAVERSE_*`),
    `--codebook-new-file`, or `parse_new_codebook_format(..., new_format_path=...)`.
    """
    d = dataset.lower()
    key = d.upper()

    def _truthy_env(name: str) -> Optional[str]:
        raw = os.environ.get(name)
        if raw is None:
            return None
        s = os.path.expandvars(str(raw).strip())
        return s if s else None

    for name in (
        f"BEHAVIOR_{key}_CODEBOOK_NEW",
        f"BEHAVIOR_{key}_CODEBOOK",
        "BEHAVIOR_CODEBOOK_NEW",
        "BEHAVIOR_CODEBOOK",
        f"DATAVERSE_{key}_CODEBOOK_NEW",
        f"DATAVERSE_{key}_CODEBOOK",
        "DATAVERSE_CODEBOOK_NEW",
        "DATAVERSE_CODEBOOK",
    ):
        cand = _truthy_env(name)
        if cand:
            env_new = cand
            break
    else:
        env_new = None

    env_old = None
    for name in (
        f"BEHAVIOR_{key}_CODEBOOK_OLD",
        "BEHAVIOR_CODEBOOK_OLD",
        f"DATAVERSE_{key}_CODEBOOK_OLD",
        "DATAVERSE_CODEBOOK_OLD",
    ):
        cand = _truthy_env(name)
        if cand:
            env_old = cand
            break

    if env_new:
        new_p = Path(os.path.expanduser(env_new)).resolve()
        old_p = Path(os.path.expanduser(env_old)).resolve() if env_old else new_p
        return new_p, old_p
    try:
        def_new, def_old = default_codebook_paths(d)
    except ValueError as exc:
        raise ValueError(
            f"No codebook path configured for slug {dataset!r}. Point to any structured codebook file "
            f"via `BEHAVIOR_{key}_CODEBOOK_NEW`, `BEHAVIOR_{key}_CODEBOOK`, shared `BEHAVIOR_CODEBOOK_NEW`, "
            "the CLI `--codebook-new-file`, or legacy `DATAVERSE_*` variables with the same meaning."
        ) from exc
    new_p = def_new
    old_p = Path(os.path.expanduser(env_old)).resolve() if env_old else def_old
    return new_p, old_p


def parse_new_codebook_format(
    dataset,
    codebook_dir=CODEBOOK_DIR,
    new_format_path: Optional[Union[str, Path]] = None,
):
    if new_format_path is not None:
        codebook_file = str(Path(new_format_path))
    else:
        codebook_file = str(resolve_codebook_paths(dataset)[0])
    with open(codebook_file, "r", encoding="utf-8") as f:
        codebook = f.read()

    #print(f"Codebook length: {len(codebook.split(' '))} whitespace words")

    instructions, codebook = codebook.split("### Categories ###")
    instructions = instructions.strip().replace("### Instructions ###", "").strip()
    instructions = instructions.split("\n")
    instruction_dict = {}
    for line in instructions:
        key, value = line.split(":", 1) # only split on the first colon
        if value.strip():
            instruction_dict[key.strip()] = value.strip()

    codebook = codebook.split("\n\n")

    codebook_list = []
    for section in codebook:
        try:
            section_list = section.strip().split("\n")
            section_dict = {"Category": section_list[0].strip()}
            for line in section_list[1:]:
                line = line.strip()
                if line.startswith("--") or not line:
                        continue
                key, value = line.split(":", 1) # only split on the first colon
                if value.strip():
                    section_dict[key.strip()] = value.strip()
        except ValueError:
            print(section)
        codebook_list.append(section_dict)
    codebook_list = [i for i in codebook_list if "Label" in i.keys()]
    return codebook_list, instruction_dict

#codebook_list, instruction_dict = parse_new_codebook_format("ccc_codebook_new_format.txt")

def make_prompt_old(model_type, system_message, user_message, answer=""):
    if model_type == "llama3":
        return make_prompt_llama3(system_message, user_message, answer)
    elif model_type == "mistral" or model_type == "llama2":
        return make_prompt_mistral_llama2(system_message, user_message, answer)
    elif model_type == "phi3":
        return make_prompt_phi3(system_message, user_message, answer)
    elif model_type == "openai":
        return make_prompt_openai(system_message, user_message, answer)
    else:
        raise ValueError("Invalid model type. Must be one of 'llama3', 'mistral', 'llama2', 'phi3', or 'openai'")

def make_prompt(tokenizer, system_message, user_message, answer=""):
    """
    Build chat prompt. Some tokenizers (e.g. Gemma 2) reject a separate system role; merge into user.
    """
    system_message = (system_message or "").strip()
    user_message = (user_message or "").strip()

    def _combined_user() -> str:
        if system_message:
            return f"{system_message}\n\n{user_message}"
        return user_message

    def _apply(messages, add_generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    try:
        if answer:
            return _apply(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{answer}"},
                ],
                add_generation_prompt=False,
            )
        return _apply(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            add_generation_prompt=True,
        )
    except jinja2.exceptions.TemplateError as exc:
        if "system" not in str(exc).lower():
            raise
        combined = _combined_user()
        if answer:
            return _apply(
                [
                    {"role": "user", "content": combined},
                    {"role": "assistant", "content": f"{answer}"},
                ],
                add_generation_prompt=False,
            )
        return _apply([{"role": "user", "content": combined}], add_generation_prompt=True)

def make_generic_message(document, codebook_list, instruction_dict, 
                         context=None,
                         meta=None,
                         source=None,
                         target=None,
                         excluded_sections=None,
                         task_order="codebook_first"
                         ):
    """
    Given an input document, codebook, and instruction dictionary, return a system message and user message.

    Parameters
    ----------
    document : str
        The input document to be categorized.
    codebook_list : list
        A list of dictionaries, each containing the category information.
    instruction_dict : dict
        A dictionary containing the instruction information.
    in_context : str
        Not implemented
    excluded_sections : list
        A list of sections to exclude from the output message. Used for ablation studies.
    task_order : str
        Either "document_first" or "codebook". Determines the order of the prompt
    """

    if excluded_sections is None:
        excluded_sections = []

    category_list = []
    for section in codebook_list:
        # get the keys that match the included_sections
        sub_section = {k: v for k, v in section.items() if k not in excluded_sections}
        sub_section = '\n'.join([f"{k}: {v}" for k, v in sub_section.items()]).strip()
        category_list.append(sub_section)
    categories = '\n\n'.join(category_list).strip()

    if context:
        context_str = f"\n\nTo provide some context for your coding, here's the previous passage: \"{context}\"."
    else:
        context_str = ""

    if meta:
        meta_str = f"\n\nThe following is {meta}."
    else:
        meta_str = ""

    if source:
        source_str = f"\n\nSource: {source}"
    else:
        source_str = ""

    if target:
        target_str = f"\nTarget: {target}"
    else:
        target_str = ""

    system_message = f"""Instructions: {instruction_dict['Instruction']}\nTask Type: {instruction_dict['Task']}"""
    if task_order == "codebook_first":
        user_message = f"""Categories:\n{categories}{context_str}{meta_str}{source_str}{target_str}\n\nDocument: {document}"""
    elif task_order == "document_first":
        user_message = f"""Document: {document}{context_str}{meta_str}{source_str}{target_str}\n\nInstructions: {instruction_dict['Instruction']}\nTask Type: {instruction_dict['Task']}\n\nCategories:\n{categories}"""
    else:
        raise ValueError("Invalid task_order. Must be one of 'document_first' or 'codebook_first'")

    if "Output Reminder" not in excluded_sections:
        user_message = user_message + "\n\n" + instruction_dict['Output Reminder']
    
    return system_message, user_message

#make_generic_message("We support them", codebook_list=codebook_list, 
#                     instruction_dict=instruction_dict, 
#                    context="The troops are important", 
#                    meta="a Canadian manifesto")

def make_generic_message_old(document, codebook, instruction_dict):
    system_message = f"Instructions: {instruction_dict['Instruction']}\n\nCategories:\n{codebook}"
    user_message = f"Document: {document}"
    return system_message, user_message



def call_llm(prompt, model, tokenizer, max_new_tokens=100,
             constrained_generation=1.0, use_custom_terminators=False):
    """
    Call the basic LLM model with the prompt, return the generated text.
    """
    terminators = []
    if use_custom_terminators:
        terminators = [tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                tokenizer.convert_tokens_to_ids("<|end|>")]
    else:
        terminators = [tokenizer.eos_token_id]
    inputs = tokenizer(prompt, return_tensors="pt",
                       return_token_type_ids=False)
    inputs = inputs.to("cuda")
    prompt_length = len(inputs['input_ids'][0])
    with torch.amp.autocast('cuda', enabled=True), torch.no_grad():
        _pad = pad_token_id_for_generate(tokenizer)
        if constrained_generation != 1.0:
            outputs = model.generate(**inputs, 
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=_pad,
                                 eos_token_id=terminators,
                                 encoder_repetition_penalty=constrained_generation)
        else:
            outputs = model.generate(**inputs, 
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=_pad,
                                 eos_token_id=terminators,
                                 do_sample=False)
    trimmed_outputs = [outputs[0][prompt_length:]]
    text = tokenizer.batch_decode(trimmed_outputs, skip_special_tokens=True)[0] 
    return text

def batch_call_llm(prompts: List[str], 
                   model, 
                   tokenizer,
                   batch_size: int = 4,
                   max_new_tokens: int = 100,
                   constrained_generation: float = 1.0,
                   use_custom_terminators: bool = False,
                   show_progress: bool = True,
                   seed: Optional[int] = None) -> List[str]:
    """
    Batched version of LLM calling with memory management and progress tracking.
    Falls back to batch_size=1 permanently after any OOM error.
    
    Args:
        prompts: List of input prompts to process
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        batch_size: Initial number of prompts to process simultaneously
        max_new_tokens: Maximum number of tokens to generate
        constrained_generation: Repetition penalty for constrained generation
        use_custom_terminators: Whether to use additional termination tokens
        show_progress: Whether to show progress bar
        seed: Random seed for reproducibility
        
    Returns:
        List of generated texts corresponding to input prompts
    """
    if len(prompts) == 0:
        return []

    # Set random seed if provided 
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Set up terminators
    terminators = [tokenizer.eos_token_id]
    if use_custom_terminators:
        terminators.extend([
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.convert_tokens_to_ids("<|end|>")
        ])

    # Initialize tokenizer settings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Pre-allocate results and track processing
    results = [None] * len(prompts)
    processed_indices = set()

    try:
        # Create iterator over batch starting indices
        indices = range(0, len(prompts), batch_size)
        if show_progress:
            indices = tqdm(indices, total=len(prompts)//batch_size + (1 if len(prompts) % batch_size else 0))

        for batch_start in indices:
            if batch_start in processed_indices:
                continue
                
            batch = prompts[batch_start:batch_start + batch_size]
            try:
                # Tokenize batch
                inputs = tokenizer(batch, 
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True,
                                 return_token_type_ids=False)
                
                # Move to GPU
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                # Padded batch: all rows share width; new tokens follow the full prompt width.
                input_width = int(inputs["input_ids"].shape[1])

                # Generate with memory optimization context
                _pad = pad_token_id_for_generate(tokenizer)
                with torch.amp.autocast('cuda', enabled=True), torch.no_grad():
                    if constrained_generation != 1.0:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=_pad,
                            eos_token_id=terminators,
                            encoder_repetition_penalty=constrained_generation
                        )
                    else:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=_pad,
                            eos_token_id=terminators,
                            do_sample=False
                        )

                # Trim and decode outputs (left-padded batch: slice at unified input width)
                for i, output in enumerate(outputs):
                    trimmed = output[input_width:]
                    text = tokenizer.decode(trimmed, skip_special_tokens=True)
                    results[batch_start + i] = text
                
                # Mark these indices as processed
                processed_indices.update(range(batch_start, batch_start + len(outputs)))
                
                # Explicit cleanup
                del inputs, outputs
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # If OOM occurs, clear memory and reprocess remaining prompts with batch_size=1
                    torch.cuda.empty_cache()
                    print(f"\nOOM error with batch_size={batch_size} at index {batch_start}.")
                    print(f"Switching to batch_size=1 for {len(prompts) - len(processed_indices)} remaining prompts...")
                    
                    # Get indices of unprocessed prompts
                    unprocessed = list(set(range(len(prompts))) - processed_indices)
                    
                    # Process remaining prompts individually
                    remaining_iterator = tqdm(unprocessed) if show_progress else unprocessed
                    for idx in remaining_iterator:
                        try:
                            single_result = batch_call_llm(
                                [prompts[idx]], 
                                model, 
                                tokenizer,
                                batch_size=1,
                                max_new_tokens=max_new_tokens,
                                constrained_generation=constrained_generation,
                                use_custom_terminators=use_custom_terminators,
                                show_progress=False,
                            )
                            results[idx] = single_result[0]
                            processed_indices.add(idx)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                raise RuntimeError("OOM error even with batch_size=1")
                            raise e
                    
                    break  # Exit main loop since we've handled remaining prompts
                else:
                    raise e

        # Verify all prompts were processed
        if None in results:
            unprocessed = [i for i, r in enumerate(results) if r is None]
            raise ValueError(f"Some prompts were not processed: indices {unprocessed}")
            
        return results
    
    finally:
        # Ensure memory is cleared even if an error occurs
        torch.cuda.empty_cache()

def batch_call_llm_old(prompts: List[str], 
                   model, 
                   tokenizer,
                   batch_size: int = 4,
                   max_new_tokens: int = 100,
                   constrained_generation: float = 1.0,
                   use_custom_terminators: bool = False,
                   show_progress: bool = True,
                   seed: Optional[int] = None) -> List[str]:
    """
    Batched version of LLM calling with memory management and progress tracking.
    Falls back to batch_size=1 permanently after any OOM error.
    
    Args:
        prompts: List of input prompts to process
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        batch_size: Initial number of prompts to process simultaneously
        max_new_tokens: Maximum number of tokens to generate
        constrained_generation: Repetition penalty for constrained generation
        use_terminators: Whether to use additional termination tokens
        show_progress: Whether to show progress bar
        seed: Random seed for reproducibility
        
    Returns:
        List of generated texts corresponding to input prompts
    """

    # Set random seed if provided 
    if seed is not None:
        # Set all relevant random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # Some models/tokenizers also use generator objects

    # Set up terminators
    terminators = [tokenizer.eos_token_id]
    if use_custom_terminators:
        terminators.extend([
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.convert_tokens_to_ids("<|end|>")
        ])

    results = []
    current_batch_size = batch_size
    
    # Create batches
    num_batches = (len(prompts) + current_batch_size - 1) // current_batch_size
    batches = [prompts[i:i + current_batch_size] for i in range(0, len(prompts), current_batch_size)]
    
    # Progress bar if requested
    batch_iterator = tqdm(batches) if show_progress else batches

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        for batch in batch_iterator:
            try:
                # Tokenize batch
                inputs = tokenizer(batch, 
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True,
                                 return_token_type_ids=False)
                
                # Move to GPU
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                input_width = int(inputs["input_ids"].shape[1])

                # Generate with memory optimization context
                _pad = pad_token_id_for_generate(tokenizer)
                with torch.amp.autocast('cuda', enabled=True), torch.no_grad():
                    if constrained_generation != 1.0:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=_pad,
                            eos_token_id=terminators,
                            encoder_repetition_penalty=constrained_generation
                        )
                    else:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=_pad,
                            eos_token_id=terminators,
                            do_sample=False
                        )
                
                # Trim and decode outputs
                batch_results = []
                for i, output in enumerate(outputs):
                    trimmed = output[input_width:]
                    text = tokenizer.decode(trimmed, skip_special_tokens=True)
                    batch_results.append(text)
                
                results.extend(batch_results)
                
                # Explicit cleanup
                del inputs, outputs
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # If OOM occurs, clear memory and reprocess all remaining prompts with batch_size=1
                    torch.cuda.empty_cache()
                    print(f"\nOOM error with batch_size={current_batch_size}. "
                          f"Switching to batch_size=1 for all remaining prompts...")

                    # Get remaining prompts but DON'T include the current batch twice
                    remaining_prompts = [p for b in batch_iterator for p in b]  # Just get the unprocessed prompts

                    # Process all remaining prompts with batch_size=1
                    new_prompts = batch + remaining_prompts
                    for prompt in tqdm(new_prompts):  # Process failed batch + remaining
                        try:
                            single_result = batch_call_llm(
                                [prompt], 
                                model, 
                                tokenizer,
                                batch_size=1,
                                max_new_tokens=max_new_tokens,
                                constrained_generation=constrained_generation,
                                use_custom_terminators=use_custom_terminators,
                                show_progress=False,
                            )
                            results.extend(single_result)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                raise RuntimeError("OOM error even with batch_size=1")
                            raise e
                    
                    # Break out of the main loop since we've processed all remaining prompts
                    if len(results) != len(prompts):
                        raise ValueError(f"Got {len(results)} results for {len(prompts)} prompts - this indicates a bug in batching!")
                    return results
                else:
                    raise e
        if len(results) != len(prompts):
            raise ValueError(f"Got {len(results)} results for {len(prompts)} prompts - this indicates a bug in batching!")
        return results
    
    finally:
        # Ensure memory is cleared even if an error occurs
        torch.cuda.empty_cache()



def parse_answers(raw_answers, allowed_labels, be_nice=True):
    """
    Given a list of raw answers, parse them into the allowed labels.

    Parameters
    ----------
    raw_answers : list
        A list of raw answers from the model.
    allowed_labels : list
        A list of allowed labels from the codebook
    be_nice : bool
        Whether to normalize before matching: replace ``\\_`` with ``_``, match labels
        case-insensitively, and return the **canonical** label string from ``allowed_labels``
        (same spelling/casing as in the codebook / gold column).

    Returns
    -------
    parsed_answers : list
        A list of parsed answers.
    """
    parsed_answers = []

    unique_labels = sorted({l for l in allowed_labels if l is not None}, key=len, reverse=True)
    if not unique_labels:
        return [None] * len(raw_answers)

    canonical_by_norm = {}
    for label in unique_labels:
        if label is None:
            continue
        k = label.replace("\\_", "_")
        k = k.upper() if be_nice else k
        if k not in canonical_by_norm:
            canonical_by_norm[k] = label

    # Regex over labels longest-first so e.g. "COOPERATIVE" vs "COOP" prefixes don't bite.
    if be_nice:
        label_regex = "|".join(
            re.escape(lbl.replace("\\_", "_").upper()) for lbl in unique_labels if lbl is not None
        )
    else:
        label_regex = "|".join(re.escape(lbl) for lbl in unique_labels if lbl is not None)

    for answer in raw_answers:
        if answer is None:
            parsed_answers.append(None)
            continue
        if be_nice:
            work = answer.replace("\\_", "_").upper()
        else:
            work = answer

        matches = re.findall(label_regex, work)
        if matches:
            key = matches[0]
            parsed_answers.append(canonical_by_norm[key])
        else:
            parsed_answers.append(None)
    return parsed_answers


def load_codebook(dataset):
    """
    Load the structured codebook and instruction dictionary for a dataset.
    The third return value remains a raw-text fallback for older prompt helpers.
    """
    d = dataset.lower()
    new_path, old_path = resolve_codebook_paths(d)
    codebook_list, instruction_dict = parse_new_codebook_format(d, new_format_path=new_path)
    if old_path.is_file():
        with open(old_path, "r", encoding='utf-8') as f:
            old_codebook = f.read()
    else:
        # Some bundles only ship the structured/new-format file (e.g. manifestos).
        with open(new_path, "r", encoding='utf-8') as f:
            old_codebook = f.read()
    return codebook_list, instruction_dict, old_codebook

