import random
import pandas as pd
from collections import deque
from typing import List, Dict
from openpyxl.workbook import Workbook
import os
import math
from collections import defaultdict
import numpy as np

# --- settings ---

n_participants = 30
base_name = "localiser_conditions" # root filename without ppt index

SEED = 7 # seed for first participant, might be better to randomise 
N_EXEMPLARS_PER_CONCEPT = 40
MISMATCH_RATIO = 0.20 # ratio of word-image mismatch pairs 
concepts = [
    "bug","berry","bicycle","bird","box","car","chair","coffee","dog","book","fish",
    "guitar","hammer","hand","house","jacket","pencil","phone","pizza","plane","tree"
]
left_key = "g" # key to choose the option presented on the left 
right_key = "b"
screen_positions = ["left", "right"] # positions on the screen with response options 


folder_name_map = {c: c for c in concepts}
german_map = {
    "bug":"Käfer","berry":"Beere","bicycle":"Fahrrad","bird":"Vogel","box":"Kiste",
    "car":"Auto","chair":"Stuhl", "coffee":"Kaffee","dog":"Hund","book":"Buch","fish":"Fisch",
    "guitar":"Gitarre","hammer":"Hammer","hand":"Hand","house":"Haus","jacket":"Jacke",
    "pencil":"Stift","phone":"Telefon","pizza":"Pizza","plane":"Flugzeug","tree":"Baum",
} # map english to german words

MAX_ATTEMPTS = 5000 # max. attempts to build trial list that matches constraints


# --- contraints for generating the condition file --- 

# constraint 01: not more than x identical concepts in a row (for words and images)
MAX_STREAK = 1 

# constraint 02: prompt the participant to indicate if wrd-img map o every 5th trial on average
# so we have total_trials / 5 = 168 prompt trials
# every concept has 168/21 = 8 prompts 
prompt_interval = 5
number_prompt_trials = int(N_EXEMPLARS_PER_CONCEPT*len(concepts)) / prompt_interval
per_concept_prompt = int(number_prompt_trials / len(concepts))

# constraint 03: try to spread the exemplars of one concept evenly 
target_per_quartile = N_EXEMPLARS_PER_CONCEPT // 4  

# constraint 04: make sure not always same images get paired with same words on mismatch trials

# constraint 05: make sure mismatches are equally enough distributed
per_concept_mismatch = int(N_EXEMPLARS_PER_CONCEPT*MISMATCH_RATIO)
MISMATCH_PER_QUARTILE = int(round((per_concept_mismatch*len(concepts)) // 4))
QUARTILE_TOL = 3  # allow deviation

# constraint 06 (in case key-resp mappings change during the experiment): 
# same button is not more than x times in a row the right answer 
MAX_STREAK_COR_KEY = 3

# --- additionally: pick ITI

# add uniform ITI within these limits 
ITI_RANGE = (0.75, 1.25) 

# --- function to find non-matching words from the list of all words --- 

# this is random choice now, should we make it more systematic?
def pick_nonmatching_word(concept: str, used_mismatches: dict, rng: random.Random) -> str:

    candidates = [w for w in concepts if w != concept]

    # constraint 04
    candidates = [w for w in candidates if w not in used_mismatches[concept]]

    # if we exhausted all options, reset memory for this concept
    if not candidates:
        used_mismatches[concept].clear()
        candidates = [w for w in concepts if w != concept]

    # pick one and remember it
    chosen = rng.choice(candidates)
    used_mismatches[concept].add(chosen)
    return chosen


# --- build valid trial list (take care of constraints) ---

def build_valid_schedule(trials: List[Dict],
                         rng: random.Random,
                         max_attempts: int,
                         max_streak: int,
                         n_exemplars_per_concept: int,
                         quartile_tol: int = 0,
                         max_prompt_streak: int = 3):
    """
    Enforces:
      - MAX_STREAK for concepts & words (constraint 01)
      - exemplar spreading: each concept ~ evenly per quartile (constraint 03)
      - prompts ~ evenly across 4 quartiles
      - no more than max_prompt_streak prompts in a row
    """
    total = len(trials)
    if total == 0:
        return []

    # For overall concept occurrences (each concept appears n_exemplars_per_concept times)
    overall_target_per_q = {}
    for c in concepts:
        base = n_exemplars_per_concept // 4
        rem  = n_exemplars_per_concept % 4
        # distribute remainder to the first 'rem' quartiles
        overall_target_per_q[c] = [base + (1 if q < rem else 0) for q in range(4)]

    # For per-concept PROMPTs (count how many prompt trials each concept has)
    prompt_count_by_concept = {c: 0 for c in concepts}
    for t in trials:
        if t.get("is_prompt", 0) == 1:
            prompt_count_by_concept[t["concept"]] += 1

    prompt_target_per_q = {}
    for c in concepts:
        tot_prompts = prompt_count_by_concept[c]
        base = tot_prompts // 4
        rem  = tot_prompts % 4
        prompt_target_per_q[c] = [base + (1 if q < rem else 0) for q in range(4)]

    # do the same as above but for mismatches
    mismatch_count_by_concept = {c: 0 for c in concepts}
    for t in trials:
        if t.get("is_mismatch", 0) == 1:
                mismatch_count_by_concept [t["concept"]] += 1

    mismatch_target_per_q = {}
    for c in concepts:
        tot_mismatches = mismatch_count_by_concept[c]
        base = tot_mismatches // 4
        rem  = tot_mismatches % 4
        mismatch_target_per_q[c] = [base + (1 if q < rem else 0) for q in range(4)]

    # attempt to build a schedule
    for _ in range(max_attempts):
        remaining = trials[:]
        rng.shuffle(remaining)
        schedule = []

        # streak trackers
        last_concepts = deque(maxlen=max_streak)
        last_words    = deque(maxlen=max_streak)

        # per-quartile counters for what we've scheduled so far
        overall_counts_q =  {c: [0, 0, 0, 0] for c in concepts}
        prompt_counts_q  =  {c: [0, 0, 0, 0] for c in concepts}
        mismatch_counts_q = {c: [0, 0, 0, 0] for c in concepts}
        prompt_streak = 0 

        ok = True
        
        for i in range(total):
            # inline of _quartile_of_index(i, total):
            # split the index range [0..total-1] into 4 bins
            q = (i * 4) // total   # 0..3

            rng.shuffle(remaining)
            picked_idx = None

            for idx, cand in enumerate(remaining):
                c = cand["concept"]
                w = cand["word_shown_english"]
                is_prompt = (cand.get("is_prompt", 0) == 1)
                is_mismatch = (cand.get("is_mismatch", 0) == 1)

                # prompt-streak cap
                if is_prompt and prompt_streak >= max_prompt_streak:
                    continue

                # (1) concept streak
                c_streak = 1
                for prev_c in reversed(last_concepts):
                    if prev_c == c:
                        c_streak += 1
                    else:
                        break
                if c_streak > max_streak:
                    continue

                # (2) word streak
                w_streak = 1
                for prev_w in reversed(last_words):
                    if prev_w == w:
                        w_streak += 1
                    else:
                        break
                if w_streak > max_streak:
                    continue

                # (3) overall concept quartile target
                if overall_counts_q[c][q] >= overall_target_per_q[c][q] + quartile_tol:
                    continue

                # (4) prompt quartile target (only if this is a prompt)
                if is_prompt:
                    if prompt_counts_q[c][q] >= prompt_target_per_q[c][q] + quartile_tol:
                        continue

                # (5) mismatches equally distributed 
                if is_mismatch:
                    if mismatch_counts_q[c][q] >= mismatch_target_per_q[c][q] + quartile_tol:
                        continue

                # passed all constraints
                picked_idx = idx
                break

            if picked_idx is None:
                ok = False
                break

            # place item and update trackers
            item = remaining.pop(picked_idx)
            schedule.append(item)

            last_concepts.append(item["concept"])
            last_words.append(item["word_shown_english"])

            overall_counts_q[item["concept"]][q] += 1
            if item.get("is_prompt", 0) == 1:
                prompt_counts_q[item["concept"]][q] += 1
                prompt_streak += 1          
            else:
                prompt_streak = 0    
                  
            if item.get("is_mismatch", 0) == 1:
                mismatch_counts_q[item["concept"]][q] += 1

        if ok and not remaining:
            return schedule

    # give up
    return []

# --- main function: build base trial list

def main(out_xlsx=base_name, seed=SEED):
    rng = random.Random(seed)
    concept_to_num = {c: i+1 for i,c in enumerate(concepts)}

    # Build base list of trials (each image once, with no constraints)
    trials = []
    for c in concepts:
        folder = folder_name_map[c]
        for ex in range(1, N_EXEMPLARS_PER_CONCEPT+1):
            img = f"{folder}\\{folder}_{ex:02d}.jpg"
            trials.append({"concept": c, "image_shown": img, "concept_num": concept_to_num[c]})
    total = len(trials)

    # get indices for prompt trials  
    prompt_index_by_concept = {c: set() for c in concepts}

    # constraint 02: prompt quota per concept 
    for c in concepts:
        # pick which exemplar indices (1..40) for this concept will be prompted
        idxs = list(range(1, N_EXEMPLARS_PER_CONCEPT+1))
        rng.shuffle(idxs)
        # pick from randomly shuffled indices which ones to prompt 
        prompt_ids = idxs[:per_concept_prompt]                 # e.g., 8 out of 40
        prompt_index_by_concept[c] = set(prompt_ids)

    mismatch_index_by_concept = {}
    mismatches_per_concept = int(round(MISMATCH_RATIO * N_EXEMPLARS_PER_CONCEPT))
    
    for c in concepts:
        idxs = list(range(1, N_EXEMPLARS_PER_CONCEPT+1))
        rng.shuffle(idxs)
        mismatch_ids = set(idxs[:mismatches_per_concept])  # 8 exemplar IDs for this concept
        mismatch_index_by_concept[c] = mismatch_ids

    # assign words to images to create mismatches 
    enriched = []

    # dict to store mismatches that are already used
    used_mismatches = {c: set() for c in concepts}

    for base in trials:
        c = base["concept"]
        # parse exemplar id from the file path (.._XX.jpg)
        ex_str = base["image_shown"].split("_")[-1].split(".")[0]
        ex_id = int(ex_str)

        prompt_idx = 1 if ex_id in prompt_index_by_concept[c] else 0
        is_mismatch = 1 if ex_id in mismatch_index_by_concept[c] else 0

        match_idx = 0 if is_mismatch == 1 else 1

        # Choose the word
        if match_idx == 1:
            w_en = c
        else:
            w_en = pick_nonmatching_word(c, used_mismatches, rng)
        try:
            w_de = german_map[w_en]
        except KeyError:
            raise KeyError(f"german_map missing entry for '{w_en}' (concept '{c}', exemplar {ex_id})")

        enriched.append({
            **base,
            "is_prompt": prompt_idx,      # keep legacy key if used elsewhere
            "prompt_idx": prompt_idx,     # explicit export
            "match_idx": match_idx,       # 1=match, 0=mismatch (independent of prompt)
            "word_shown_english": w_en,
            "word_shown_german": w_de,
        })
        
    # try building schedule
    sched = build_valid_schedule(
        enriched,
        rng=rng,
        max_attempts=MAX_ATTEMPTS,
        max_streak=MAX_STREAK,
        n_exemplars_per_concept=N_EXEMPLARS_PER_CONCEPT,
        quartile_tol=QUARTILE_TOL
    )
    if not sched:
        raise RuntimeError(
        "build_valid_schedule() returned no trials. "
        "Try increasing MAX_ATTEMPTS or relaxing constraints (e.g., quartile_tol=1)."
    )
    # add ITI and response_side mapping 
    rows = []
    correct_button_storage = deque(maxlen=MAX_STREAK_COR_KEY)

    for t in sched: # loop trials 

        iti = rng.uniform(*ITI_RANGE)

        # pick a side that doesn't violate the correct-button streak rule
        while True:
            nonmatch_pres_side = rng.choice(screen_positions)

            # correct button 
            # compute the correct response for THIS TRIAL
            if t["match_idx"] == 0:  # mismatch: nonmatch side is correct
                resp = left_key if nonmatch_pres_side == "left" else right_key
            else:                    # match: opposite side is correct
                resp = left_key if nonmatch_pres_side == "right" else right_key

            streak_counter = 1
            for past_corr in reversed(correct_button_storage): 
                if past_corr == resp: 
                    streak_counter+=1
                else: break # stop streak counter for mismatch            
            if streak_counter <= MAX_STREAK_COR_KEY:
                break 
        
        correct_button_storage.append(resp)

        # append missing rows 
        rows.append({
            "concept": t["concept"],
            "concept_num": t["concept_num"],
            "image_shown": t["image_shown"],
            "word_shown_english": t["word_shown_english"],
            "word_shown_german": t["word_shown_german"],
            "prompt_idx": t["prompt_idx"],        
            "match_idx": int(t["match_idx"]),          
            "ITI_length": round(iti, 2),
            "correct_response": resp,
            "nonmatch_pres_side": nonmatch_pres_side,
        })

    # to df 
    df = pd.DataFrame(rows)
    
    n_mismatch = int((df["match_idx"] == 0).sum())
    n_prompts = int(df["prompt_idx"].sum())
    # export
    df.to_excel(out_xlsx, index=False)
    print(f"Wrote {out_xlsx} with {len(df)} trials, prompts={n_prompts}, mismatches={n_mismatch}.")


if __name__ == "__main__":
    import argparse, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_participants", type=int, default=30)
    ap.add_argument("--out_dir", default = r"c:\sync_folder\TSRlearn-task\sequences") 
    args = ap.parse_args()

    base_name = "localiser_conditions"
    os.makedirs(args.out_dir, exist_ok=True)

    for ppt in range(args.n_participants):
        ppt_id = f"{ppt:02d}"
        out_xlsx = os.path.join(args.out_dir, f"{base_name}_{ppt_id}.xlsx")
        main(out_xlsx=out_xlsx, seed=SEED + ppt)  

