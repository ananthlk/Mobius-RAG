# Path B: tag matching, rejected codes, and new-tag extraction

## 1. Logic of matching to tags

**Source:** `app/services/policy_path_b.py` — `get_phrase_to_tag_map`, `_apply_tags_to_line_text`

**Flow:**

1. **Lexicon shape**  
   The lexicon is loaded from `policy_lexicon_meta` + `policy_lexicon_entries`. Each **approved** tag has:
   - `kind`: `p` (prescriptive), `d` (descriptive), or `j` (jurisdiction)
   - `code`: tag identifier (e.g. `member_eligibility`)
   - `spec`: JSONB that can include a **`phrases`** list (normalized strings to match in text)

2. **Phrase map**  
   `get_phrase_to_tag_map(lexicon_snapshot)` builds a single map:
   - **Key:** normalized phrase (lowercase, whitespace collapsed) from every `spec.phrases` entry across all p/d/j tags
   - **Value:** `(kind, code)` for that tag  
   If the same phrase appears in more than one tag, the last one in iteration wins (no duplicate keys).

3. **Line normalization**  
   For each policy line, the line text is normalized the same way: lowercase, collapse runs of whitespace to a single space, strip.

4. **Matching rule**  
   For each `(phrase, (kind, code))` in the phrase map:
   - If **`phrase in normalized_line`** (substring match), the line gets that tag with score **1.0**.
   - The tag is stored in the line’s `p_tags`, `d_tags`, or `j_tags` JSONB as `{ code: 1.0 }`.

**Important:** Only **approved** lexicon entries (in `policy_lexicon_entries` with `active = true`) are used. There is no “rejected code” in the lexicon; rejected items are only in the **candidate catalog** and affect **extraction**, not matching.

**Summary:**  
Matching is **substring-based**: normalized line text is checked for each approved phrase; if the phrase appears anywhere in the line, that line is tagged with the corresponding (kind, code) at score 1.0.

---

## 2. How rejected codes are handled

**“Rejected”** here means: a **candidate** (a proposed phrase) was reviewed and set to state **rejected** by a human. That is not a “code” in the lexicon; it’s a decision about a candidate.

**Where it’s stored:**

- **`policy_lexicon_candidates`**  
  Rows have `state = 'rejected'` (or `approved` / `proposed` / `flagged`).
- **`policy_lexicon_candidate_catalog`**  
  When you review a candidate (approve/reject/proposed/flagged), the API upserts a catalog row with `(candidate_type, normalized_key, proposed_tag_key, state)`. So the catalog is the **global** record of “this normalized phrase was rejected (or approved, etc.)”.

**How they’re used:**

1. **Tag matching**  
   Rejected candidates are **not** in the lexicon. So they are **never** used when matching line text to tags. Only approved entries in `policy_lexicon_entries` are used.

2. **Candidate extraction (not resurfacing)**  
   When we **extract new candidates** (see below), we should **not** propose again a phrase that was already rejected. So:
   - We load the set of **rejected** normalized phrases from `policy_lexicon_candidate_catalog` (where `state = 'rejected'`).
   - Any n-gram whose normalized form is in that set is **skipped** and not inserted as a new `PolicyLexiconCandidate`.

So: **rejected “codes” (phrases) are handled by** (1) never being in the lexicon (so never matched as tags), and (2) being excluded from candidate extraction so they don’t resurface as proposed candidates.

---

## 3. Process to extract new tags (candidates)

**Source:** `app/services/policy_path_b.py` — `extract_candidates_for_document`

**Goal:** Find spans of text in policy lines that are **not** in the approved lexicon (and not already rejected) and create **policy_lexicon_candidates** for human review.

**Steps:**

1. **Inputs**
   - All atomic **policy_lines** for the document.
   - **phrase_map**: normalized phrase → (kind, code) from the **approved** lexicon (same as used for tag matching).
   - **Rejected set**: normalized phrases from `policy_lexicon_candidate_catalog` where `state = 'rejected'` (so we don’t re-propose them).

2. **N-gram collection**
   - For each line, normalize the line text (same as tag matching).
   - Split into words and collect **word n-grams** for **n = 2, 3, 4** (sliding window).
   - For each n-gram, **normalize** it (lowercase, collapse spaces) to get a key.

3. **Filtering**
   - **Skip** if the n-gram key is in **phrase_map** (already an approved phrase).
   - **Skip** if the n-gram key is in the **rejected set** (already rejected).
   - **Skip** if length &lt; 4 characters.
   - Count how many times each remaining n-gram appears (across all lines).

4. **Insert candidates**
   - Keep only n-grams with **count ≥ min_occurrences** (default 2).
   - Cap total new candidates per document (e.g. 200).
   - For each kept n-gram, insert a **PolicyLexiconCandidate** row with:
     - `document_id`, `run_id`
     - `candidate_type = 'd'` (descriptive)
     - `normalized` = the n-gram
     - `proposed_tag` = n-gram with spaces → underscores, lowercased
     - `confidence` = f(count), e.g. `min(0.9, 0.5 + 0.1 * count)`
     - `source = 'path_b_ngram'`
     - `occurrences` = count
     - `state = 'proposed'`

5. **After review**
   - Human reviews in the UI and sets state to **approved** or **rejected** (or proposed/flagged).
   - On **approve**, the API calls `approve_phrase_to_db`, which adds the phrase to **policy_lexicon_entries** (so it becomes an approved tag and is used in future tag matching).
   - On **reject**, the API upserts **policy_lexicon_candidate_catalog** with `state = 'rejected'`, so the next Path B run will not re-extract that phrase as a candidate.

**Summary:**  
New tags are found by collecting 2–4 word n-grams from policy lines, dropping those already in the lexicon or in the rejected catalog, and creating proposed candidates for n-grams that appear at least `min_occurrences` times; after human approval they are added to the lexicon and used in tag matching; after rejection they are only used to suppress future re-proposal.
