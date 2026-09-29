# PROMPT — KYC dossiers: RAW OCR → structured extraction → reference matching → conformity
### Qwen3.6-27B-FP8 · NVIDIA H100 · Domino Data Lab · Hugging Face Transformers

---

## 0. Role

You are a **Senior AI/ML Engineer, Senior Multimodal LLM Inference Engineer, Senior OCR/Document AI Engineer, KYC/AML compliance data engineer, PyTorch/Hugging Face expert, NVIDIA H100 performance engineer, and production-grade AI pipeline architect.**

---

## 1. Context and the attached baseline notebook

In addition to this prompt, I am attaching the notebook:

```text
kyc_qwen36_raw_ocr_pipeline.ipynb
```

This is the notebook with which we **successfully extracted the complete raw text (RAW OCR) of every page of every target document of one customer dossier**, with excellent quality (French, Arabic, English, MRZ lines, handwriting), running in our Domino Data Lab environment on the H100 with the local `Qwen3.6-27B-FP8` checkpoint.

**This notebook was produced with GPT Astra 6.** It descends from an earlier notebook, `dom.ipynb` (a different business case, domiciliation documents), whose Qwen3.6 / FP8 inference foundations it reuses.

It is now the **empirically validated baseline and the primary technical reference** for this work:

* inspect it **completely** before writing any code;
* keep what works — its inference path is proven in our environment;
* inspect it critically: any bug, scale problem or weakness you find must be fixed **and** explained;
* any deviation from it must be justified in a Markdown cell next to the changed code.

### What I need now

Evolve the attached notebook into a production pipeline that:

1. processes **ALL customer dossiers in the ZIP — imperatively** (≈ 200 dossiers, ≈ 1 000 target documents);
2. applies a new **duplicate rule based on file size**;
3. keeps producing the **raw OCR first** (the raw text is the source of truth), then **extracts structured information from the raw text**, with document-specific rules;
4. attaches to every extracted value a **probabilistic confidence computed from the model**, a **manual-review flag**, and whether the value is **handwritten or machine-printed**;
5. writes the extracted information **per document, in JSON files, organised per customer dossier**;
6. **matches** the extracted information against our reference file `recertication_lissage_updated.csv`, flags every inconsistency, and states whether **each processed document** and **each dossier** is conforme;
7. is **optimised** to process ≈ 200 dossiers / ≈ 1 000 documents on one H100.

### Deliverable

Return one complete, downloadable, executable Jupyter notebook:

```text
kyc_qwen36_extraction_conformite_pipeline.ipynb
```

with the pipeline and its detailed implementation steps (§16).

---

## 2. Target architecture

```text
STAGE 0   ZIP → safe extraction → inventory → size/SHA-256 dedup → reference CSV check
          → page plan (tiers) → capacity plan                                        [CPU]
STAGE 1   RAW OCR: 1 page = 1 Qwen call (UNCHANGED from the attached notebook)
          + passive token-confidence capture (no effect on the output)                [GPU]
STAGE 2A  Deterministic extraction from raw_ocr: document markers, label anchors,
          MRZ (ICAO 9303) with check digits, dates, names, account numbers            [CPU]
STAGE 2B  Visual verification + handwritten/printed: 1 short call per page carrying
          extracted fields; optional text-only fallback for a missing field           [GPU]
STAGE 3   Matching with recertication_lissage_updated.csv → document conformity
          → dossier conformity, flags                                                 [CPU]
STAGE 4   JSON per document per dossier, dossier JSON, reports, manual-review queue   [CPU]
```

Governing principles:

* `raw_ocr` is never modified. Everything downstream is **derived**, **traceable** to a character span of a page's `raw_ocr`, and **re-runnable without the GPU** (§11.4).
* The GPU is used only where the model is indispensable (Stage 1, Stage 2B). Business rules, matching and reporting are CPU-only and fast.
* One H100, one model load per kernel, batch size 1 by default.

---

## 3. What stays EXACTLY as in the attached notebook (Stage 1 foundations)

The attached notebook's Stage 1 produced the validated results. Keep the following **as they are** (same names, same arguments, same behaviour), extending them only where this prompt explicitly asks:

| Area | Keep from the attached notebook |
|---|---|
| Configuration | `@dataclass(frozen=True) class Config` / `CFG` pattern — **extend** it, do not replace it |
| Packages | Package validation without modifying Domino (`RECOMMENDED_MINIMUMS`, `INSTALLED_VERSIONS`, no automatic installation). Add optional detection of `rapidfuzz` and `openpyxl` — never required |
| Persistence helpers | `json_safe`, `json_text`, `sha256_file`, `stable_hash`, `atomic_text`, `atomic_json`, `csv_text`, `record_error` |
| Environment | CUDA / H100 / BF16 diagnostics cell, environment and package logs |
| ZIP | `normalized_filename`, `logical_document`, `safe_member_path`, `safe_extract_zip()` with manifest, hash-keyed extraction directory, Zip-Slip / zip-bomb / encryption / collision guards, `ZIP_CUSTOMER_ROOT`, wrapper detection; only target PDFs are materialised |
| Inventory | `build_inventory` (extended with dedup columns, §6.1) |
| Rendering | `render_page` (zoom 3.0 → max side 1 400; HD zoom 4.5 → max side 2 200; `MAX_RENDER_PIXELS`; `colorspace=fitz.csRGB`), `image_statistics`, `is_blank`, `prepare_image`, `ROTATION_OVERRIDES`; all preprocessing switches **False** by default; `import pymupdf as fitz` with `fitz` fallback |
| Model loading | `AutoProcessor.from_pretrained(..., trust_remote_code=True, local_files_only=True, min_pixels=CFG.MIN_PIXELS, max_pixels=CFG.MAX_PIXELS)`, `processor.tokenizer.padding_side = "left"`, `AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True, quantization_config=FP8Config(dequantize=True), local_files_only=True)`, `model.eval()`, the local FP8 `config.json` verification, the installed-quantizer inspection, `MODEL_SIGNATURE`, the "model already loaded in this kernel" guard |
| Placement proof | `model_diagnostics()` — **blocking** on CPU/disk/meta offload, leftover float8 parameters, missing BF16, non-CUDA input embeddings; `INPUT_DEVICE`, `GPU_DEVICES`, `cuda_sync`, `gpu_memory`, `cleanup_document`, `recover_cuda_context` |
| OCR prompt | `RAW_OCR_PROMPT` — **byte-for-byte unchanged** |
| Template | `make_messages`, `apply_template` (strict `TypeError` fallback), `inspect_template()` (open-`<think>` block check, switch-effect check) |
| Output validation | `looks_like_mrz`, `detect_degenerate_output`, `validate_transcription` and its statuses (`SUCCESS`, `BLANK_PAGE`, `DEGENERATE`, `THINKING_OUTPUT`, `TRUNCATED`, `TIME_LIMIT`, `UNEXPECTED_STOP`, `SUSPICIOUS_SHORT`, `FORMAT_VIOLATION`, `CUDA_OOM`, `ERROR`), `DiagnosticStop` (time limit + repetition guard), `FirstStepLogitsAudit` |
| Generation | `infer_image`: `processor(text=..., images=..., return_tensors="pt")`, `pixel_values` / `image_grid_thw` / image-token checks, `inputs.to(INPUT_DEVICE)`, `model.generate(**inputs, max_new_tokens=CFG.MAX_NEW_TOKENS_RAW_OCR, do_sample=False, num_beams=1, repetition_penalty=1.0, use_cache=True, pad_token_id=pad_id, stopping_criteria=..., logits_processor=..., return_dict_in_generate=False, output_scores=False)`, stop-reason logic, input-token trimming, decode with `skip_special_tokens=True, clean_up_tokenization_spaces=False`, special-token decode for `<think>` detection |
| Orchestration | `run_attempt`, `save_attempt`, `select_attempt`, `save_page_record`, `page_identity`, `PIPELINE_FINGERPRINT`, atomic per-page checkpoints, per-attempt records, `export_results`, `CallBudget`, cold/warm benchmark gate, `hd_is_appropriate`, `process_page` / `process_pdf` / `process_customer` structure, `print_summary`, `show_performance` |
| Defaults | rendering values above, `MIN_PIXELS = 4*32*32`, `MAX_PIXELS = 2600*32*32`, `MAX_NEW_TOKENS_RAW_OCR = 2048`, `MAX_GENERATION_SECONDS = 120`, `ENABLE_HD_FALLBACK = False`, `SKIP_BLANK_PAGES = False` |

**Why "byte-identical" matters:** because the Stage 1 inputs, prompt and decoding do not change, re-running the already-validated dossier must reproduce its previous `raw_ocr`. That gives us a **regression oracle** (§7.6) proving that none of the new work degraded the OCR.

---

## 4. Carry-over requirements from the previous brief (still binding)

These requirements from the brief that produced the attached notebook remain in force:

* **Keep the working Qwen3.6 / FP8 / H100 foundations.** Do not change `AutoProcessor`, `AutoModelForImageTextToText`, `FineGrainedFP8Config(dequantize=True)`, `dtype=torch.bfloat16`, `device_map="auto"`, the processor invocation, the chat-template construction, `enable_thinking=False`, or the input-token trimming without evidence — and explain exactly why if you do.
* Do not introduce `AutoModelForCausalLM`, `Qwen2_5_VLForConditionalGeneration`, vLLM, TensorRT-LLM or another framework "because it might be faster".
* Do not re-quantize the already-FP8 checkpoint. Do not use a legacy `FP8Linear` import.
* Do not blindly reinstall PyTorch, CUDA, Transformers or Accelerate inside Domino. Detect installed versions first; install only a missing non-core package, and only if necessary.
* Load the model **once** per kernel. Never reload it per dossier or per PDF.
* Deterministic decoding (`do_sample=False`, `repetition_penalty=1.0`); no meaningless sampling-parameter combinations.
* **One primary Qwen call per OCR page.** A fallback call only when validation fails. No multi-crop strategies, no classification call — the logical document type is already known from the filename.
* Measure every stage independently (render, preprocessing, template, processor, device transfer, generation, decode, validation, checkpoint I/O), with `torch.cuda.synchronize()` at GPU boundaries, so slowness can be attributed to a specific cause.
* Keep the **performance gate**: benchmark one page cold then warm **before** any large run; print projections; block on an abnormal configuration.
* Keep the `!!!!!!!!` degeneration detection, MRZ-safe (valid MRZ `<` runs are never flagged).
* Page-level checkpoints keyed by customer id, logical type, physical relative path, PDF SHA-256, page number and pipeline fingerprint; atomic writes; safe resume; failed/degenerate results are never silently reused.
* Error isolation at every level (ZIP, customer, PDF, page, render, preprocessing, processor, transfer, generation, decode, validation, checkpoint write, and now extraction, matching and reporting). CUDA OOM: record, release, continue only if the CUDA context is usable. Never hide exceptions.
* `gc.collect()` / `torch.cuda.empty_cache()` per document, not per page (except after an OOM).
* **Stage 1 output stays raw:** never translate, transliterate, normalise, correct or re-order the stored `raw_ocr`. Normalised views are allowed in Stage 2 only, always with a map back to raw offsets.
* JSON serialisation safe for NumPy / PyTorch / Path / NaN values; UTF-8 everywhere; IDs always strings.
* Code quality: real executable Python; no pseudocode; no `pass` / `TODO` on critical paths; no undefined helpers; `pathlib`; type hints where useful; docstrings; `logging`; dataclasses; explicit configuration; defensive checks; inference logic kept visible and debuggable; no gratuitous object-oriented complexity.
* The notebook must run cell by cell in Domino Data Lab.

**Lifted from the previous brief:** "no structured extraction / no database comparison / no confidence" is no longer true. Stage 2 and Stage 3 are now in scope — **strictly downstream** of the raw OCR, which stays Stage 1.

---

## 5. Input data

### 5.1 ZIP archive (unchanged)

```text
kyc_documents.zip
├── <customer_id>/            one folder per customer; folder name = customer id
│   ├── JUSTIFICATIF IDENTITE.PDF
│   ├── JUSTIFICATIF IDENTITE(1).PDF
│   ├── JUSTIFICATIF DOMICILE.PDF
│   ├── CONVENTION COMPTE.PDF
│   ├── FATCA.PDF
│   ├── CARTON SIGNATUTE.PDF
│   └── other_documents.pdf   (ignored)
└── ...
```

### 5.2 Target documents (unchanged)

```python
TARGET_DOCUMENTS = (
    "JUSTIFICATIF IDENTITE.PDF",
    "JUSTIFICATIF DOMICILE.PDF",
    "CONVENTION COMPTE.PDF",
    "FATCA.PDF",
    "CARTON SIGNATUTE.PDF",      # historical business spelling: stays the canonical key
)
DOCUMENT_ALIASES = {"CARTON SIGNATURE.PDF": "CARTON SIGNATUTE.PDF"}
```

Filename matching stays **controlled** (case, whitespace, numeric suffix `(n)`, explicit alias). No fuzzy filename matching: an unrelated PDF must never be mapped onto one of the five regulated categories.

### 5.3 Reference file (new)

```text
recertication_lissage_updated.csv      (UTF-8; keep this exact filename in the configuration)
```

It contains what was entered in our system for all customers:

| Column | Meaning | Format |
|---|---|---|
| `Id tiers` | customer id = dossier folder name = `customer_id` in the pipeline | text — read as string, never as integer (leading zeros) |
| `Date de naissance` | date of birth | `yyyy-mm-dd` |
| `Nom abrege tiers` | nom + prénom, or prénom + nom, separated by a space | text |
| `Date expiration document` | expiry date of the identity document | `yyyy-mm-dd` |
| `Numero de compte` | account number — matched against the one extracted from CARTON SIGNATURE | text |

Loading requirements:

* `encoding="utf-8-sig"` (tolerates a BOM), `dtype=str`, `keep_default_na=False`; auto-detect the separator (`,`, `;` or tab) and print it.
* Header matching tolerant to accents, case and spacing (`Nom abrégé tiers` = `Nom abrege tiers`, `Numéro de compte` = `Numero de compte`). If a required column is missing, fail loudly and print the headers found.
* Strip whitespace. Parse dates strictly as `yyyy-mm-dd` (tolerate a trailing ` 00:00:00`). An invalid reference value becomes a data-quality flag, not a crash.
* One `Id tiers` may appear on several rows (e.g. several accounts): group by id. Name, date of birth and expiry should agree across rows — flag `REFERENCE_INCONSISTENT_ROWS` otherwise. Account numbers form a set.
* **Every dossier must exist in the base:** its folder name must be present in `Id tiers`. Exact match after `strip()`; a leading-zero-insensitive fallback match is allowed but flagged `ID_MATCH_LEADING_ZEROS`. Not found → `DOSSIER_ABSENT_BASE`. The dossier is **still OCR'd and extracted**.
* Report both directions: dossiers in the ZIP absent from the CSV, and CSV ids without a dossier (informational).
* This check runs in Stage 0, **before** the model is loaded, so inconsistencies are visible immediately.

---

## 6. Stage 0 changes

### 6.1 Duplicate rule based on file size (new)

**Business rule.** Inside a customer folder, several physical files can correspond to the same document type with slightly different names, for example:

```text
JUSTIFICATIF IDENTITE.PDF
JUSTIFICATIF IDENTITE(1).PDF
JUSTIFICATIF IDENTITE(2).PDF
```

* If these files have the **same size**, run the OCR on **only one** of them.
* If their sizes **differ**, run the OCR on **all** of them.

Implementation:

1. Grouping uses the attached notebook's `logical_document()` mapping. Add a unit test proving that suffixes without a space (`IDENTITE(1).PDF`) and with a space (`IDENTITE (1).PDF`) are both recognised.
2. Within each `(customer_id, logical_document_type)` group, sub-group files by **exact byte size**.
3. Same size → confirm with the SHA-256 already computed at extraction. Same size **and** same hash = identical copy: OCR one representative (deterministic choice: the name without suffix, else the lowest `(n)`, else lexicographic order); record the others as `SKIPPED_IDENTICAL_COPY` with `representative_path`; the representative's results apply to them.
4. Same size but **different** SHA-256 (rare) means different content: the size-only rule would silently drop a different document. Default: OCR both and flag `SAME_SIZE_DIFFERENT_CONTENT`. Config `DEDUP_REQUIRE_SHA256_MATCH = True`; setting it to `False` applies the literal size-only rule.
5. Different sizes → OCR all (`PROCESSED_DISTINCT_VERSION`).
6. Inventory gains `size_group_id`, `dedup_decision` ∈ {`SINGLE`, `REPRESENTATIVE`, `SKIPPED_IDENTICAL_COPY`, `PROCESSED_DISTINCT_VERSION`}, `representative_path`, `identical_copies`.
7. This replaces `DUPLICATE_POLICY` (`"all"` / `"error"`) for the production run; keep `"error"` only as a diagnostic option.

How several **distinct** processed files of one logical document are consolidated is defined in §10.5.

### 6.2 Page plan in tiers (new)

The business rules (§8) use: **all pages** of JUSTIFICATIF IDENTITE (the MRZ is often on the back side), **page 1 only** of JUSTIFICATIF DOMICILE and CONVENTION COMPTE, **all pages** of FATCA (the two forms may be on different pages) and CARTON SIGNATURE.

* **Tier 1** (default, run first for every dossier): exactly the pages the rules use.
* **Tier 2** (backfill, optional): every remaining page, OCR'd **after** tier 1 has completed for all dossiers, with the same Stage 1 code and checkpoints. `RUN_TIER2_BACKFILL = False` by default. This produces the complete raw-text archive when GPU time allows, without delaying the business results.
* `OCR_PAGE_POLICY = "REQUIRED_FIRST"` (default) or `"ALL_PAGES"` (tiers 1 and 2 in a single pass).
* **Dynamic escalation:** if page 1 of DOMICILE or CONVENTION is `BLANK_PAGE` or lacks the document marker and the PDF has more pages, OCR page 2 in tier 1 (at most one extra page), flagged `PAGE1_NOT_USABLE_ESCALATED`.
* Safety cap `MAX_TIER1_PAGES_BY_TYPE` for pathological PDFs (e.g. a 15-page identity file): pages beyond the cap move to tier 2 and the document is flagged.

### 6.3 Capacity plan and call budget

Before the production run, print and save:

* dossiers, physical PDFs, identical copies skipped, distinct versions, missing documents;
* pages per document type and per tier; pages already checkpointed;
* planned OCR calls, planned Stage 2B calls, fallback reserve, benchmark calls;
* per-document-type warm time measured on the pre-flight dossier, ETA for tier 1, tier 2 and Stage 2B;
* the derived `MAX_QWEN_CALLS` (planned × safety factor, printed). The existing `CallBudget` stays authoritative: no call is ever made beyond it.

---

## 7. Stage 1 changes (the OCR itself does not change)

### 7.1 Passive token-confidence recorder (new)

The confidence of an extracted value must be **computed from the model**. The only legitimate source is the model's own token probabilities during the OCR generation.

* Add `TokenConfidenceRecorder(transformers.LogitsProcessor)`, appended **after** `FirstStepLogitsAudit` in the same `LogitsProcessorList`.
* In Transformers 4.57, custom processors are appended **after** the internal default processors, the logits are cast to float32 before any processor runs, and greedy decoding takes `argmax` of the fully processed scores. The recorder therefore sees exactly the distribution each token was chosen from. **Verify this in the installed version** (inspect `_merge_criteria_processor_list`, `_get_logits_processor`, `_sample`) and print the evidence in the notebook.
* Per step, record on the GPU: the log-softmax values and token ids of the top-k (k = 5). **Return `scores` unchanged** — no in-place operation.
* No `.item()` / `.cpu()` per step (that forces a GPU sync per token). Accumulate small tensors and transfer them once after `generate`.
* Do **not** use `output_scores=True` or `output_logits=True`: they keep a vocabulary-sized float32 tensor per step (≈ 150 k × 4 bytes × 2 048 steps ≈ 1.2 GB per page).
* After `generate`: assert `recorded_steps == tokens_out` and that the recorded argmax id equals the generated id at every step. A mismatch is a hard error for that attempt (`CONFIDENCE_CAPTURE_MISMATCH`).
* Storage: sidecar `outputs/token_confidence/<page_key>.json.gz` with, per token, `token_id`, `char_start`, `char_end`, chosen-token log-probability, top-1/top-2 margin; top-5 alternatives only for tokens with probability below `ALT_STORE_BELOW_P` (e.g. 0.9). The page checkpoint gets a summary (mean and min token probability, share of tokens with p < 0.5).
* The recorder must not change the output — proven by §7.6.
* Add the recorder's schema version to `PIPELINE_FINGERPRINT` and bump `PIPELINE_VERSION`: pages without token data must be re-OCR'd. This is expected (only one dossier has been processed so far).

### 7.2 Token → character offset map (new)

* Map every generated token to `[char_start, char_end)` in the stored `raw_ocr` (decoded with `skip_special_tokens=True, clean_up_tokenization_spaces=False`).
* Use incremental decoding in the style of `transformers.TextStreamer`: decode from the last stable boundary, hold back while the text ends with U+FFFD (an incomplete multi-byte character — frequent with Arabic), reset at newlines. Complexity O(n·L). Never re-decode the whole prefix at every token (O(n²)).
* Assert that the concatenation of the pieces equals `raw_ocr`. If it does not, align the two strings with `difflib` and flag `OFFSET_MAP_ALIGNED`.
* Special tokens map to empty spans.

### 7.3 Attempt finality and retry policy (changed)

The attached notebook retries every non-reusable page at every resume. At ≈ 1 000 documents this wastes GPU time on persistent content issues (stamps, photos, dense pages).

* `MAX_ATTEMPTS_PER_PAGE` (default 2 per pipeline fingerprint, HD included).
* Once attempts are exhausted the page gets a **final** status (`TRUNCATED_FINAL`, `SUSPICIOUS_SHORT_FINAL`, `DEGENERATE_FINAL`, …) and is not retried at resume unless listed in `FORCE_RETRY_STATUSES`.
* `TRUNCATED`: exactly one retry with `MAX_NEW_TOKENS_RETRY` (e.g. 4 096) and a proportionally raised `MAX_GENERATION_SECONDS_RETRY` (otherwise the 120 s cooperative deadline of `DiagnosticStop` would cut the retry short), only if the repetition guard did not fire; then final.
* Stage 2 uses the best available `raw_ocr` even when the page is not `SUCCESS` (a truncated page often still contains the fields), and marks the affected fields for review (`OCR_STATUS_<STATUS>`).

### 7.4 Circuit breaker at scale (changed)

`STOP_AFTER_CONSECUTIVE_BAD_PAGES = 2` would abort a multi-hour run because of two consecutive `SUSPICIOUS_SHORT` pages. In production mode, stop only on **systemic** failure:

* ≥ `SYSTEMIC_CONSECUTIVE_FAILURES` (default 6) consecutive attempts in {`DEGENERATE`, `THINKING_OUTPUT`, `UNEXPECTED_STOP`, `FORMAT_VIOLATION`, generation/decode `ERROR`} spanning at least 2 distinct PDFs; or
* a failure rate above `SYSTEMIC_FAILURE_RATE` (default 30 %) over the last `SYSTEMIC_WINDOW` (default 50) attempts; or
* fatal CUDA or persistence errors (unchanged).

Content statuses (`BLANK_PAGE`, `SUSPICIOUS_SHORT`, `TRUNCATED`) never stop the run. Keep the strict original breaker for the pre-flight dossier.

### 7.5 Page images at scale (changed)

`SAVE_PAGE_IMAGES = True` writes two full-resolution PNGs per attempt: thousands of files, many gigabytes, and PNG encoding of a ~4.5-megapixel render costs a noticeable fraction of a second each. Production default: `SAVE_PAGE_IMAGES = "REVIEW_ONLY"` — a JPEG (quality 85, max side 1 400) of the prepared image, only for pages whose status is not `SUCCESS` or whose document needs manual review. The pre-flight may keep `"ALL"`.

### 7.6 Regression oracle (new, mandatory before the production run)

* `PREFLIGHT_CUSTOMER_ID`: the dossier already validated. If `None`, reuse the attached notebook's selection rule (seed 42 over the sorted customers with at least one extracted target PDF) so the pre-flight lands on the same dossier.
* `PREVIOUS_RAW_OCR_JSONL`: path to the previous run's `outputs/raw_ocr/raw_ocr_results.jsonl`.
* Compare page by page (key: `customer_id`, `customer_relative_path`, `page_number`, `pdf_sha256`): exact-match rate, normalised character similarity, and a diff for every page that differs.
* Pass criterion: exact match on ≥ 95 % of pages and similarity ≥ 0.99 on every page (BF16 GPU kernels are not guaranteed bitwise-deterministic). Otherwise stop and print the diffs before the production run.
* If the previous JSONL is not available, skip with a clear warning.

---

## 8. Stage 2A — structured extraction from `raw_ocr` (business rules)

All information is extracted **from the raw OCR text**. The document labels ("metadata" such as `Nom`, `اللقب`, `Surname`) can be written in French, Arabic or English on an Algerian (DZ) document, and in any language on a foreign document.

The business rules below are the user's rules and must be implemented **exactly**. The paragraphs marked *Implementation* add the engineering precision needed to apply them reliably to OCR text; they never change a rule.

### 8.0 Shared text toolkit

**Normalised views** (derived only — `raw_ocr` is never modified):

* *Latin view:* NFKC; upper case; accents removed (NFKD, combining marks dropped); quotes `« » “ ” „ " ' ’ ‘` unified; dashes `‐ ‑ – — −` → `-`; whitespace collapsed; zero-width and bidi control characters removed (U+200B–U+200F, U+202A–U+202E, U+2066–U+2069, U+FEFF).
* *Arabic view:* tashkeel removed (U+064B–U+065F, U+0670); tatweel U+0640 removed; `أ إ آ ٱ` → `ا`; `ى` → `ي`; `ة` → `ه`; `ؤ` → `و`; `ئ` → `ي`; whitespace collapsed.
* *Digits:* Arabic-Indic (U+0660–U+0669) and Extended Arabic-Indic (U+06F0–U+06F9) → ASCII, for parsing only.
* Every view carries an **index map back to raw offsets**: any match found on a view yields a raw span `[start, end)` and a page number. This is what links a value to its token probabilities (§9.1).

**Marker detection** (document-type evidence):

1. Exact match of the normalised marker in the normalised view → `MARKER_EXACT`.
2. Otherwise an OCR-tolerant match: sliding-window similarity (`rapidfuzz.fuzz.partial_ratio` if installed, else a `difflib` implementation) ≥ `MARKER_FUZZY_THRESHOLD` (default 90) → `MARKER_FUZZY`. Weaker evidence: the document goes to review; it is never silently treated as exact.
3. Arabic markers tolerate the optional definite article `ال` and flexible spacing (`شهادة إقامة` ≈ `شهادة الإقامة`).
4. Long titles may wrap across lines: match on a whitespace-collapsed view.
5. Record the evidence: marker, method, score, page, raw span.

**Label-anchored value extraction:**

* Labels are regex alternatives over the normalised views.
* Value = the text after the label on the same line (after `:`, `/` or `-`), otherwise the next non-empty line(s) that are not themselves labels.
* For Arabic labels, right-to-left rendering can put the value **before** the label on the same line (e.g. `1974/08/12 :تاريخ الميلاد`) — check both sides.
* For passport labels, the value is on the line(s) **below** the label line.
* Type-check candidates (name, date, account number); keep alternatives; ambiguity lowers the extraction confidence.

**Date parser** `parse_date(raw_text, role)`:

* Formats: `DD.MM.YYYY`, `DD/MM/YYYY`, `DD-MM-YYYY`, `DD MM YYYY`, `YYYY.MM.DD`, `YYYY/MM/DD`, `YYYY-MM-DD`, `DD MMM YYYY` with French and English month names or abbreviations (`12 AOU 1974`, `12 AUG 1974`, `12 AOÛT 1974`), bilingual forms (`12 AOU/AUG 74`), two-digit years (role-based century), Arabic-Indic digits.
* Algerian convention is day-first. If both day and month are ≤ 12 and no MRZ corroborates → `DATE_DAY_MONTH_AMBIGUOUS` (review).
* Returns the ISO date, the detected format, the ambiguity flag and the raw span.

**MRZ parser (ICAO Doc 9303):**

* Detect candidate lines (reuse `looks_like_mrz`) and assemble TD1 (3 × 30), TD2 (2 × 36) and TD3 (2 × 44) blocks. Tolerate spaces inserted by the OCR. Length repair only by adding/removing `<` fillers, accepted only if every check digit validates (`MRZ_LENGTH_REPAIRED`).
* Check digit: weights 7-3-1 repeating; `0-9` = digit value, `A-Z` = 10–35, `<` = 0; sum mod 10. Validate document number, date of birth, expiry, TD3 personal number, composite.
* Constrained correction: in numeric-only positions, try the usual OCR confusions (`O→0`, `D→0`, `Q→0`, `I→1`, `L→1`, `Z→2`, `S→5`, `G→6`, `B→8`) **only** when the substitution makes the check digit pass → `MRZ_CORRECTED` (review).
* Parse: document code, issuing state, nationality, surname and given names (`<<` separates them, `<` → space), date of birth, sex, expiry, document number.
* Centuries: date of birth → the most recent century that keeps it ≤ `REFERENCE_DATE`; expiry → `20YY` unless that exceeds `REFERENCE_DATE` + 30 years, then `19YY`.
* MRZ names are transliterated and may be truncated (TD1 name field 30 characters, TD3 39): compare visual zone ↔ MRZ on accent-free upper case, spaces and hyphens → `<`, with prefix tolerance.
* Test vectors — ICAO specimens, all check digits verified valid:

```text
TD3  P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<
     L898902C36UTO7408122F1204159ZE184226B<<<<<10
     document L898902C3 → 6 | birth 740812 → 2 | expiry 120415 → 9
     personal ZE184226B<<<<< → 1 | composite → 0
     composite = line2[0:10] + line2[13:20] + line2[21:43]

TD1  I<UTOD231458907<<<<<<<<<<<<<<<
     7408122F1204159UTO<<<<<<<<<<<6
     ERIKSSON<<ANNA<MARIA<<<<<<<<<<
     document D23145890 → 7 | birth 740812 → 2 | expiry 120415 → 9 | composite → 6
     composite = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]

Negative test: birth field "74O812" (letter O) must fail its check digit (2).
```

**`REFERENCE_DATE`** = the run date (`date.today()`, Africa/Algiers), overridable in the configuration for reproducibility, and written into every output.

---

### 8.1 `JUSTIFICATIF IDENTITE.PDF`

#### A. Algerian documents

For Algerians, the document must be a **national identity card**, a **driving licence** or a **passport**.

**1. Algerian document.** The document is Algerian if its `raw_ocr` contains:

```text
الجمهورية الجزائرية الديمقراطية الشعبية
```

or `DZA`, or `DZ`.

*Implementation:* `DZ` / `DZA` must be matched as **tokens** (e.g. `(?<![A-Z0-9])DZA?(?![A-Z0-9])` on the Latin view). A plain substring test would classify any foreign document carrying a name such as `DZIEDZIC` or `RODZIK` as Algerian. The strongest evidence is the MRZ issuing state / nationality field equal to `DZA`. Record the evidence; if the evidence conflicts (Algerian header but foreign MRZ nationality) → `NATIONALITY_CONFLICT` (review).

**2. Biometric check — the FIRST check once the document is known to be Algerian** (before any structuring). The document must be biometric. It is detected by the presence of the MRZ zone (machine-readable lines containing symbols such as `<`).

* Not biometric → flag `NOT_BIOMETRIC` and **do not proceed to structuring** (no field extraction). The document is `NON_CONFORME`. Keep `raw_ocr`, the detected type and the evidence.

**3. Document type.**

| Type | `raw_ocr` contains |
|---|---|
| National identity card (carte d'identité nationale) | `بطاقة التعريف الوطنية` |
| Passport | `PASSPORT/PASSEPORT` **or** `جواز السفر` |
| Driving licence (permis de conduire) | `DRIVING LICENCE` **or** `رخصة السياقة` |

*Implementation:* `PASSPORT/PASSEPORT` tolerates spaces around `/` and either order. The MRZ document code corroborates (`P…` = passport, `I…` / `ID` = identity card). No type found → `UNKNOWN_IDENTITY_DOCUMENT_TYPE` (`NON_CONFORME`). Several types found → `DOCUMENT_TYPE_AMBIGUOUS` (review; the MRZ document code decides when present).

**Algerian regulation and formats.** Refer to the Algerian biometric document formats, which follow ICAO Doc 9303: the biometric national identity card is an ID-1 card with a **TD1** MRZ (3 × 30, issuing state `DZA`); the biometric passport has a **TD3** MRZ (2 × 44, `P<DZA…`). Do **not** hard-code legal validity durations or any regulatory fact you cannot cite; make such checks configurable.

*Open point to report, not to change silently:* it has not been confirmed that every Algerian biometric driving licence carries an MRZ. Implement the rule as stated (MRZ required for all three types), make it configurable per subtype (`BIOMETRIC_REQUIRES_MRZ_BY_SUBTYPE`), and report the MRZ detection rate per subtype in the pre-flight and final reports so the business can decide.

**4. Fields — only if the document is biometric.** Four values: **nom, prénom, date d'expiration du document, date de naissance**.

*National identity card* — take the values of these fields present in `raw_ocr`, which represent respectively the name, first name(s), expiry date and date of birth:

```text
Nom:
Prénom(s):
:تاريخ الانتهاء
:تاريخ الميلاد
```

*Passport* — take the values located **below** each of these labels, which represent respectively the name, first name(s), date of birth and expiry date:

```text
اللقب/ Surname/ Nom
الاسم/ Given names/Prénom
مولود في/ Date of birth/ Date de naissance
ينتهي في/ Date of expiry/ Date d'expiration
```

(match a label line by any of its sub-labels; tolerate spacing and slash variations)

*Driving licence* — take the values of these fields, which represent respectively the name, first name, date of birth and expiry date:

```text
1.
2.
3.
:تاريخ الانتهاء
```

*Implementation:* tolerate `1 .`, `1)`; field 3 may be followed by the place of birth; accept `4b.` as an alternative expiry label; restrict numbered-field parsing to the licence context.

*MRZ cross-check* (all three types): compare names, date of birth and expiry with the MRZ. Agreement raises the confidence (§9.1); disagreement → `MRZ_VIZ_MISMATCH` (review). If a visual-zone field is missing but the MRZ provides it with a valid check digit, use the MRZ value (`source = "MRZ"`).

**5. Expiry.** For all three types: if the expiry date is earlier than the current date (`REFERENCE_DATE`) → flag the document `DOCUMENT_EXPIRED`.

#### B. Foreign (non-Algerian) documents

* A document is **non-Algerian** if it contains neither `DZA` nor `DZ` (token-matched as above) nor the Algerian header.
* For foreigners the document **must be a passport**: check that `raw_ocr` contains the term "passport" in **any language**. If it is not a passport → flag `FOREIGN_NOT_PASSPORT` (`NON_CONFORME`).
  *Implementation:* an extendable multilingual list (`PASSPORT`, `PASSEPORT`, `PASAPORTE`, `PASSAPORTE`, `PASSAPORTO`, `REISEPASS`, `PASPOORT`, `PASZPORT`, `PASAPORT`, `CESTOVNÍ PAS`, `ÚTLEVÉL`, `ΔΙΑΒΑΤΗΡΙΟ`, `ПАСПОРТ`, `جواز سفر`, `گذرنامه`, `护照`, `旅券`, `여권`, `पासपोर्ट`, …); an MRZ with document code `P` is language-independent, strong evidence.
* Make sure the passport is **not expired**. If expired → flag `DOCUMENT_EXPIRED` and **extract the expiry date**. By default also extract the remaining fields for traceability (`EXTRACT_ALL_FIELDS_WHEN_EXPIRED = True`); the document stays `NON_CONFORME`.
* If everything is OK, extract: **nom (Latin script), prénom (Latin script), date de naissance, date d'expiration du document, and the MRZ field if it exists** (full MRZ lines).
  *Implementation:* labels can be in any language, so the MRZ is the primary source (language-independent); a multilingual visual-zone label dictionary is secondary; the text-only fallback (§9.4) is the last resort.
* No biometric gate for foreign passports (the MRZ requirement is stated for Algerian documents only); `mrz_present` is still recorded.

#### C. Reference comparison (Stage 3)

nom + prénom ↔ `Nom abrege tiers`; date de naissance ↔ `Date de naissance`; date d'expiration ↔ `Date expiration document`.

---

### 8.2 `JUSTIFICATIF DOMICILE.PDF`

Always in accordance with Algerian regulation.

**Algerian client** (nationality taken from the dossier's identity document, §8.6). The document can be a residence card or a residence certificate:

| Type | `raw_ocr` contains |
|---|---|
| Carte de résidence | `بطاقة الإقامة` |
| Certificat de résidence | `شهادة إقامة` |

* In both cases, take from `raw_ocr` the **character string written in Latin letters at the bottom of page 1**. This string is the holder's **name and first name in Latin letters** (confirmed by the business).
  *Implementation:* scan page-1 lines bottom-up; skip empty, `[UNREADABLE]`, digit-only and punctuation-only lines; take the bottom-most line (or block of up to 2 consecutive lines) whose letters are ≥ 80 % Latin script; keep the raw span.
* Neither marker → `WRONG_DOCUMENT` (`NON_CONFORME`).
* Reference comparison: the Latin name ↔ `Nom abrege tiers`.

**Foreign client.** `raw_ocr` must contain:

```text
بطاقة المقيم الأجنبي
```

* If not → flag `NON_CONFORME_FOREIGN_RESIDENCE` (`NON_CONFORME`).
* If it does → extract the **name and first name in Latin letters**; compare with `Nom abrege tiers`.

*Implementation:* test the foreign-resident marker **before** the residence-card marker, so that the specific phrase is never shadowed by the generic one. If the nationality is unknown (identity document missing or undetermined): `بطاقة المقيم الأجنبي` present → foreign rules, otherwise Algerian rules; flag `NATIONALITY_INFERRED_FROM_DOMICILE`.

---

### 8.3 `CONVENTION COMPTE.PDF`

* Check that it is the right document on **page 1**: page 1's `raw_ocr` contains `CONVENTION DE COMPTE PARTICULIER`.
* If so, extract from the `raw_ocr` of that **same page** the name and first name in Latin letters, in the field:

```text
- Nom & Prénom :
```

*Implementation:* tolerate `Nom & Prenom`, `Nom et Prénom`, an optional leading dash and spacing around `:`; value on the same line, else on the next line.

* Marker absent from page 1 → `WRONG_DOCUMENT` (`NON_CONFORME`). If page 1 is blank or unusable and the marker is found on page 2 through the escalation of §6.2 → `MARKER_NOT_ON_PAGE_1` (`A_VERIFIER`).
* Reference comparison: name ↔ `Nom abrege tiers`.

---

### 8.4 `FATCA.PDF`

The document can contain two types of forms:

| Form | Present if `raw_ocr` contains |
|---|---|
| A | `FORMULAIRE D'IDENTIFICATION « US-PERSON» AU REGARD DE LA LOI FATCA SOUSCRIPTEUR/ASSURE` |
| B | `FATCA Group-Central Team` |

* For each form type that exists, extract from `raw_ocr` the **name and first name** if they are filled in.

*Implementation:* form A tolerates quote and apostrophe variants (`« »`, `"`, `'`, `’`), spacing inside `« US-PERSON»`, case, accents (`ASSURÉ`) and line wrapping; form B tolerates case, hyphen and spacing variants. Locate each form's page(s). Name labels come from a FR/EN dictionary (`Nom`, `Prénom`, `Nom et prénom`, `Nom & Prénom`, `Name`, `Last name`, `First name`, `Surname`, `Given name(s)`, …). Print the FATCA `raw_ocr` in the pre-flight so the anchors can be confirmed on real forms. Filled-in values are often handwritten.

* Neither form → `WRONG_DOCUMENT` (`NON_CONFORME`). Form(s) found but no name filled in → `FATCA_NAME_NOT_FILLED` (`A_VERIFIER`). Every filled-in name ↔ `Nom abrege tiers`.

---

### 8.5 `CARTON SIGNATURE.PDF` (canonical key `CARTON SIGNATUTE.PDF`)

* Check that it is the right document: `raw_ocr` contains `SPECIMEN DE SIGNATURE` (*implementation:* tolerate `SPÉCIMEN`).
* If so, extract from `raw_ocr`:
  * the value of the field `Nom & Prénom:`;
  * the value located **below** the field `Numéro de compte` (*implementation:* tolerate `Numero de compte`, `N° de compte`; value = next non-empty line(s); keep raw; digits-only normalised form for matching).
* **Conformity rule:** CARTON SIGNATURE is conforme if **at least one** of the two extracted fields matches the base: name ↔ `Nom abrege tiers` **or** account number ↔ `Numero de compte`.

---

### 8.6 Dossier context and cross-document checks

* The nationality is determined once per dossier from `JUSTIFICATIF IDENTITE` (best-evidence file, §10.5) and drives the `JUSTIFICATIF DOMICILE` rules.
* Informational check: names across documents should be consistent (token-set comparison) → `CROSS_DOCUMENT_NAME_INCONSISTENT` (informational by default; configurable to affect conformity).

---

## 9. Stage 2B — confidence, handwritten vs printed, manual review

Every extracted value must carry: a **probabilistic confidence computed from the model**, whether it **needs a manual re-check**, and whether it is **handwritten or machine-printed**.

### 9.1 Field-level confidence (model-derived, reproducible)

* `ocr_confidence` = geometric mean of the chosen-token probabilities over the tokens overlapping the value's raw span, i.e. `exp(mean(logprob))`; also record the minimum token probability and the minimum top-1/top-2 margin. Any `[UNREADABLE]` inside the span → 0.
* `extraction_confidence`: 1.0 for an unambiguous exact label anchor; documented lower constants for fuzzy label/marker matches or multiple candidates; for text-only fallback values (§9.4), the probability of the fallback's generated value tokens (same recorder), and only after substring verification.
* Corroboration as independent evidence: when the MRZ agrees with a valid check digit, combine `1 − (1 − p_visual_zone) × (1 − p_mrz)`; the visual verification of §9.2 enters the same way (or as a documented multiplicative factor — choose, justify, keep it explicit).
* `overall_confidence` ∈ [0, 1], computed by a single function whose formula is printed in the notebook.
* **Forbidden:** asking the model to state a confidence number (`"confidence": 0.95`). Self-reported scores are not probabilities.
* `extraction_confidence` and the reference-match score (§10) are different quantities — never mix them.

### 9.2 Handwritten vs printed + visual verification — one short call per page carrying fields

* **Keep `RAW_OCR_PROMPT` untouched.** Do not add handwriting markers to it: that would change the validated output and destroy the regression oracle.
* Instead, for each page that carries at least one extracted value, make **one short VLM call** with the same prepared page image and a short prompt listing the values. Strict output, one line per value:

```text
<index>|<V or N>|<P or H or M>
V = visible on the page exactly as written, N = not visible or different
P = machine-printed, H = handwritten, M = mixed or stamped
```

* Same message/template path as the OCR (`make_messages`-style, `enable_thinking=False`), deterministic decoding, small `max_new_tokens` (≈ 8 × number of values + 16), same guards, recorder active: the probability of each label = probability of the chosen label token at that step (store the top alternatives).
* Per value: `writing_type` ∈ {`printed`, `handwritten`, `mixed`, `unknown`} + `writing_type_probability`; `visual_check` ∈ {`confirmed`, `not_confirmed`, `unknown`} + probability.
* Page images are not kept at scale (§7.5), so Stage 2B re-renders the page with the same deterministic `render_page` + `prepare_image`. Record `prepared_image_sha256` in every Stage 1 attempt, and require the re-rendered image to have the same hash (mismatch → `VERIFICATION_IMAGE_MISMATCH`, values marked `unknown` + review).
* Checkpointed per page with its own fingerprint (values + prompt + image hash), counted in `CallBudget`. A parse failure yields `unknown` + review, never a crash.
* `ENABLE_VISUAL_VERIFICATION = True`; its cost appears in the capacity plan (about one short call per page carrying fields).

### 9.3 Manual-review rules

`needs_manual_review` + `review_reasons` per field and per document. Review is required when any of these holds:

* `overall_confidence` < `REVIEW_CONFIDENCE_THRESHOLD` (default 0.90, recalibrated in §14);
* value not found, or `[UNREADABLE]` in its span;
* `MRZ_VIZ_MISMATCH`, `MRZ_CORRECTED`, MRZ check-digit failure;
* `DATE_DAY_MONTH_AMBIGUOUS`;
* handwritten value with a low writing/visual probability, or `visual_check = not_confirmed`;
* document marker matched only fuzzily;
* reference comparison `PROBABLE_MATCH`, or reference value missing;
* OCR page not `SUCCESS` (`TRUNCATED_FINAL`, `SUSPICIOUS_SHORT_FINAL`, …);
* value obtained through the text-only fallback.

### 9.4 Optional text-only fallback (`ENABLE_LLM_EXTRACTION_FALLBACK = True`)

* Triggered **only** when the document marker was found but a required field is still missing after deterministic extraction.
* Text-only call on the already loaded model (no image): input = the page `raw_ocr` + the list of missing fields; output = strict JSON `{field: exact substring or null}`.
* **Anti-hallucination:** reject any value that is not a verbatim substring of `raw_ocr` (after whitespace normalisation). Accepted values get their raw span and OCR confidence like any other value, `source = "LLM_FALLBACK"`, and go to review.
* Verify during the pre-flight that text-only generation works with this processor/model class (no `images` argument).
* Checkpointed and budgeted.

---

## 10. Stage 3 — matching with `recertication_lissage_updated.csv` and conformity

For each processed file of a dossier, say whether it is conforme to the base. Then: if **all processed files** are conforme, the **dossier** is conforme; otherwise flag everything that is not conforme.

### 10.1 Name matching (`Nom abrege tiers` = nom + prénom or prénom + nom, space-separated)

* Normalise both sides (Latin view, punctuation removed, whitespace collapsed).
* `MATCH` if the token multisets are equal (order-insensitive), or equal once spaces are removed (compound names `BEN ALI` / `BENALI`, `AIT-AHMED` / `AIT AHMED`).
* The reference is an *abbreviated* name (nom abrégé): if the reference's last token is a strict prefix (≥ 3 characters) of the corresponding extracted token → `MATCH` with the note `REFERENCE_NAME_TRUNCATED` (configurable); initials (`M` vs `MOHAMED`) → `PROBABLE_MATCH`.
* `PROBABLE_MATCH` if fuzzy token-set similarity ≥ `NAME_PROBABLE_THRESHOLD` (default 90; `rapidfuzz` if installed, otherwise `difflib`) — covers OCR errors and transliteration variants (`MOHAMED` / `MOHAMMED`).
* `MISMATCH` otherwise; `NOT_EXTRACTED` and `REFERENCE_MISSING` are separate outcomes.
* Store the reference value, extracted value, both normalised forms, method, score and result.

### 10.2 Dates

Equal ISO dates → `MATCH`; day/month swapped equals the reference → `PROBABLE_MATCH` (`DAY_MONTH_SWAPPED`); otherwise `MISMATCH`.

### 10.3 Account number

Digits only. Equal → `MATCH`; difference limited to leading zeros → `MATCH` (noted); one contained in the other with ≥ 10 digits (e.g. account number vs full 20-digit RIB) → `PROBABLE_MATCH` (`ACCOUNT_PARTIAL`); otherwise `MISMATCH`. Compare against every account of the customer's rows. Do not invent a RIB-key algorithm; add one only if the bank provides its specification.

### 10.4 Document conformity: `CONFORME` / `NON_CONFORME` / `A_VERIFIER`

| Document | `NON_CONFORME` if | `A_VERIFIER` if | `CONFORME` if |
|---|---|---|---|
| JUSTIFICATIF IDENTITE | unknown/wrong type; DZ not biometric; foreign document not a passport; expired; name, birth date or expiry `MISMATCH` | any `PROBABLE_MATCH`, missing field, or undecidable condition (rules below) | not expired, biometric (DZ) or passport (foreign), and name, birth date and expiry all `MATCH` |
| JUSTIFICATIF DOMICILE | DZ: neither marker; foreign: `بطاقة المقيم الأجنبي` absent; name `MISMATCH` | `PROBABLE_MATCH`, `NOT_EXTRACTED`, undecidable condition | marker present and name `MATCH` |
| CONVENTION COMPTE | marker absent from page 1; name `MISMATCH` | `MARKER_NOT_ON_PAGE_1`, `PROBABLE_MATCH`, `NOT_EXTRACTED` | marker on page 1 and name `MATCH` |
| FATCA | no form found; a filled-in name `MISMATCH` | no name filled in; `PROBABLE_MATCH` | ≥ 1 form found and every filled-in name `MATCH` |
| CARTON SIGNATURE | marker absent; name **and** account `MISMATCH` | no `MATCH` but ≥ 1 `PROBABLE_MATCH`, or fields not extracted | name `MATCH` **or** account `MATCH` |

For every document: dossier absent from the base → comparisons impossible → `NON_CONFORME` (`DOSSIER_ABSENT_BASE`).

Evaluation rules for this table:

* **Precedence:** `NON_CONFORME` conditions first, then `CONFORME`, otherwise `A_VERIFIER` (conformity cannot be decided automatically: `PROBABLE_MATCH`, `NOT_EXTRACTED`, `REFERENCE_MISSING`, fuzzy-only marker, `DOCUMENT_TYPE_AMBIGUOUS`, `NATIONALITY_CONFLICT`, `MISMATCH_LOW_CONFIDENCE`).
* **A mismatch is only as certain as the reading behind it.** A `MISMATCH` makes the document `NON_CONFORME` only if the extracted value's `overall_confidence` ≥ `REVIEW_CONFIDENCE_THRESHOLD`. A `MISMATCH` on a low-confidence, not-visually-confirmed or handwritten value becomes `A_VERIFIER` with the flag `MISMATCH_LOW_CONFIDENCE`: an OCR misread cannot be excluded, and declaring it non-conforme would produce false alarms.
* **An exact match corroborates the reading.** Low OCR confidence alone never downgrades a `MATCH`; record an `INFO` note instead.
* `needs_manual_review` is reported independently of conformity: a `CONFORME` document may still be queued for review for informational reasons.

### 10.5 Several distinct processed files for one logical document

Evaluate each file. `MULTI_FILE_POLICY = "BEST_EVIDENCE"` (default): the logical document takes the best status (`CONFORME` > `A_VERIFIER` > `NON_CONFORME`); ties → for identity, the latest expiry, otherwise the highest mean confidence. Always list every file with its status and flags (`ADDITIONAL_FILE_NON_CONFORME`). Alternative: `"ALL_MUST_CONFORM"`.

### 10.6 Dossier conformity

* `CONFORME` if **every processed document** is `CONFORME`.
* `NON_CONFORME` if at least one processed document is `NON_CONFORME`, or the dossier is absent from the base — with the list of what is not conforme.
* `A_VERIFIER` if no document is `NON_CONFORME` but at least one needs manual verification.
* Missing documents are reported **separately**: `completeness = COMPLET | INCOMPLET` + list of missing types (`MISSING_DOCUMENT_POLICY = "SEPARATE"` by default, following the rule "all **processed** files"; `"NON_CONFORME"` available).

### 10.7 Flag taxonomy

One closed, documented enumeration: `code`, `severity` (`BLOCKING` / `REVIEW` / `INFO`), `stage`, `description`. Every flag written anywhere must belong to it (asserted at write time).

---

## 11. Stage 4 — outputs

### 11.1 Layout

```text
outputs/
├── inventory/          kyc_document_inventory.csv|json (+ dedup columns), reference_check.csv
├── raw_ocr/            raw_ocr_results.jsonl|csv|txt           (Stage 1 exports, unchanged)
├── token_confidence/   <page_key>.json.gz
├── checkpoints/        pages, attempts, verification/, llm_fallback/
├── dossiers/
│   └── <customer_id>/
│       ├── JUSTIFICATIF_IDENTITE.json
│       ├── JUSTIFICATIF_DOMICILE.json
│       ├── CONVENTION_COMPTE.json
│       ├── FATCA.json
│       ├── CARTON_SIGNATUTE.json
│       └── _dossier.json
├── reports/
│   ├── conformite_dossiers.csv        one row per dossier
│   ├── conformite_documents.csv       one row per logical document
│   ├── champs_extraits.csv            one row per extracted field
│   ├── flags.csv                      one row per flag
│   ├── revue_manuelle.csv             prioritised manual-review queue
│   ├── calibration_confiance.csv
│   └── rapport_conformite_kyc.xlsx    same sheets (only if openpyxl is installed)
├── performance/
├── logs/
└── images/             review-only JPEGs
```

### 11.2 Per-document JSON (example: Algerian identity card)

```json
{
  "schema_version": "kyc_document_v1",
  "pipeline_version": "…",
  "rules_version": "…",
  "reference_date": "2026-09-29",
  "customer_id": "123456",
  "logical_document_type": "JUSTIFICATIF IDENTITE.PDF",
  "physical_files": [
    {"customer_relative_path": "JUSTIFICATIF IDENTITE.PDF", "size_bytes": 482113,
     "pdf_sha256": "…", "dedup_decision": "REPRESENTATIVE", "identical_copies": ["JUSTIFICATIF IDENTITE(1).PDF"],
     "pages": [{"page_number": 1, "ocr_status": "SUCCESS", "page_key": "…", "mean_token_probability": 0.981}]}
  ],
  "selected_evidence_file": "JUSTIFICATIF IDENTITE.PDF",
  "classification": {
    "nationality": "DZ",
    "nationality_evidence": [{"marker": "الجمهورية الجزائرية الديمقراطية الشعبية", "method": "MARKER_EXACT", "page": 1, "span": [0, 39]},
                             {"marker": "MRZ issuing state DZA", "method": "MRZ", "page": 2}],
    "document_subtype": "CARTE_IDENTITE_NATIONALE",
    "subtype_evidence": [{"marker": "بطاقة التعريف الوطنية", "method": "MARKER_EXACT", "page": 1}],
    "is_biometric": true,
    "mrz": {"format": "TD1", "lines": ["…", "…", "…"], "check_digits": {"document": true, "birth": true, "expiry": true, "composite": true},
            "corrections": []}
  },
  "fields": {
    "nom": {
      "raw_value": "BENALI", "normalized_value": "BENALI", "source": "VIZ_LABEL",
      "label": "Nom:", "page": 1, "char_span": [212, 218],
      "ocr_confidence": 0.993, "min_token_probability": 0.97, "extraction_confidence": 1.0,
      "corroboration": ["MRZ_AGREES"], "overall_confidence": 0.9998,
      "writing_type": "printed", "writing_type_probability": 0.99,
      "visual_check": "confirmed", "visual_check_probability": 0.98,
      "needs_manual_review": false, "review_reasons": []
    },
    "prenom": {"…": "…"},
    "date_naissance": {"raw_value": "12.08.1974", "normalized_value": "1974-08-12", "…": "…"},
    "date_expiration": {"raw_value": "…", "normalized_value": "2031-05-03", "…": "…"}
  },
  "checks": {"is_expired": false},
  "reference_match": {
    "dossier_in_reference": true,
    "comparisons": {
      "nom_prenom": {"reference": "BENALI MOHAMED AMINE", "extracted": "BENALI MOHAMED AMINE", "method": "TOKEN_SET", "score": 100, "result": "MATCH"},
      "date_naissance": {"reference": "1974-08-12", "extracted": "1974-08-12", "result": "MATCH"},
      "date_expiration": {"reference": "2031-05-03", "extracted": "2031-05-03", "result": "MATCH"}
    }
  },
  "document_conformity": "CONFORME",
  "needs_manual_review": false,
  "flags": [],
  "timings": {"ocr_generation_s": 0.0, "stage2_s": 0.0}
}
```

### 11.3 `_dossier.json`

`customer_id`, `in_reference` (+ match method), `nationality`, per-document `{conformity, needs_manual_review, flags}`, `dossier_conformity`, `non_conforming_items` (document + reason), `completeness`, `missing_documents`, `review_reasons`, `pipeline_version`, `rules_version`, `reference_date`.

### 11.4 Output rules

* All JSON through `json_safe` / `atomic_json`, UTF-8 with `ensure_ascii=False` (Arabic stays readable), stable key order. CSVs in UTF-8 with BOM for Excel; identifiers as strings.
* Stage 2/3/4 outputs are **regenerated from checkpoints**. A `STAGE2_ONLY = True` mode re-runs rules, matching and reporting **without loading the model** (cached Stage 2B results are reused; missing ones → `unknown` + review). Business rules must be iterable in minutes without re-OCR.
* Separate versions: `PIPELINE_VERSION` (Stage 1) and `RULES_VERSION` (Stage 2–4). Changing a rule never invalidates the OCR.

---

## 12. Performance — ≈ 200 dossiers / ≈ 1 000 documents on one H100

Decode speed of a 27 B model dequantised to BF16 is bounded by memory bandwidth, so wall-clock time is dominated by **how many pages are generated and how many tokens each produces**. Optimise in this order, and measure each step:

1. **Do not OCR what the rules do not need:** tier-1 page plan (§6.2), identical-copy dedup (§6.1), only target PDFs extracted (already the case).
2. **Do not waste GPU on failures:** keep the repetition guard and the time limit; attempt finality (§7.3), no retry of final pages at resume; systemic-only circuit breaker (§7.4).
3. **Remove CPU/disk overhead from the page loop:** review-only images (§7.5); compute each SHA-256 once and cache it by `(path, size, mtime)`; rebuild the aggregated exports every N dossiers and at the end, not after every page.
4. **Keep the GPU for what needs it:** Stage 2A, 3 and 4 run on the CPU in milliseconds per document; Stage 2B and fallbacks are short, budgeted and checkpointed. Schedule GPU work contiguously (all of Stage 1, then all of Stage 2B) rather than interleaving it with CPU stages.
5. **O(1) memory per step for confidence capture** (§7.1); never `output_scores`.
6. **Per-type `max_new_tokens` only if data-driven:** derived from the observed `tokens_out` distribution (e.g. p99 × 1.5, never below the largest legitimate page observed); otherwise keep 2 048.
7. **CPU/GPU overlap only if measured:** if render + preprocessing + image write + processor exceed ~5 % of page time, add a prefetcher. PyMuPDF is not thread-safe: either one dedicated worker thread is the only code touching `fitz` during the run, or a process pool created **before** model loading whose workers never touch CUDA. Justify with measurements.
8. **Batching stays optional and off by default** (`ENABLE_EXPERIMENTAL_BATCHING = False`). It may be enabled only after an equivalence test on the pre-flight dossier (batch vs batch 1: identical extracted fields and per-page character similarity ≥ 0.995), with buckets by document type and image size to limit padding waste, per-sequence recorder / guards / stop reasons, and a fall-back to batch 1 for a bucket after an OOM.
9. Model loaded once; `empty_cache` per document only.
10. **Progress and ETA** per dossier (pages done, calls used, rolling tokens/s, ETA), printed and logged.

**Domino execution:** a multi-hour production run should be launched as a **Domino Job** (headless execution of the notebook), because an interactive workspace can time out. Checkpoints make a restart resume automatically. Outputs must go to persistent Domino storage (dataset or mounted volume), not to ephemeral workspace disk.

---

## 13. Notebook structure

0. Title; changelog vs the attached notebook (kept / modified / added + why); architecture; how to run (pre-flight → production; Domino Job; `STAGE2_ONLY`)
1. Package validation (+ optional `rapidfuzz`, `openpyxl`)
2. Imports
3. Configuration — extended frozen dataclass; `RUN_SCOPE = "ALL_DOSSIERS"` by default, gated by notebook sections 16, 17 and 22 below (benchmark, regression oracle, pre-flight)
4. Persistence helpers, environment / H100 diagnostics
5. Safe ZIP extraction
6. Inventory + size / SHA-256 dedup
7. Reference CSV loading, validation, dossier ↔ base check
8. **GPU-free self-tests of all business logic** (must pass before the model is loaded)
9. Rendering and preparation
10. Model loading + placement proof (skipped when `STAGE2_ONLY`)
11. OCR prompt + template inspection
12. Output validation, guards, token-confidence recorder, offset map (+ real-tokenizer offset test)
13. Instrumented inference, attempts, finality policy
14. Checkpoints, fingerprints, exports
15. Page plan (tiers), capacity plan, call budget
16. Benchmark gate
17. Pre-flight dossier — Stage 1 + regression oracle
18. Stage 2A extraction engine (toolkit, MRZ, dates, per-document rules)
19. Stage 2B visual verification + writing type + optional text-only fallback
20. Stage 3 matching, conformity, flag taxonomy
21. Stage 4 writers (per-document JSON, dossier JSON, reports, review queue)
22. Pre-flight dossier — Stages 2–4 end-to-end, printed for human review
23. Production — Stage 1 tier 1 for **all dossiers** (resumable, systemic breaker, progress / ETA)
24. Production — Stage 2A for all dossiers (CPU)
25. Production — Stage 2B for all dossiers (GPU, checkpointed)
26. Production — confidence finalisation, Stage 3, Stage 4 (CPU; also the `STAGE2_ONLY` entry point)
27. Optional tier-2 backfill
28. Calibration and quality analysis
29. Performance analysis (stage breakdown, as in the attached notebook, + Stage 2 timings)
30. Final summary — business (dossiers in base, conformes, non conformes, à vérifier, complets / incomplets, documents per status, top flags, review-queue size) and technical (calls, tokens, time per stage)
31. Status and flag glossary

---

## 14. Tests and verification

* **GPU-free self-tests** (assertions, run before loading the model): filename normalisation and dedup (`(1)` with and without a space, same size + same hash, same size + different hash, different sizes); Latin and Arabic normalisation with offset maps; marker detection (exact, fuzzy, optional `ال`); `DZ` token boundary (negative case `DZIEDZIC`); date parser (every format, day/month ambiguity, Arabic-Indic digits); MRZ (ICAO vectors of §8.0, negative test, constrained correction); label extraction on synthetic `raw_ocr` for each case — DZ identity card, DZ passport, DZ driving licence, DZ non-biometric, foreign passport, foreign non-passport, domicile DZ card, domicile DZ certificate, domicile foreign, convention, FATCA A, FATCA B, carton; name / date / account matching (truncation, compound names, initials, day/month swap, partial account); conformity tables of §10.4; dossier aggregation; flag enumeration.
* **Offset-map test with the real tokenizer** (processor loaded): French, Arabic and MRZ samples → encode → incremental decode → concatenation equals full decode.
* **Recorder test** on the benchmark page: steps = `tokens_out`, argmax = generated ids.
* **Regression oracle** (§7.6).
* **Pre-flight end-to-end** on `PREFLIGHT_CUSTOMER_ID`: print each page's `raw_ocr`, every extracted field with its span, confidences, writing type, reference comparison, document and dossier conformity. The production cells refuse to run if the benchmark, the regression oracle or the pre-flight failed, unless an explicit, logged override is set.
* **Calibration** (§28 of the notebook): with the reference agreement as weak labels (`MATCH` vs `MISMATCH` where a reference value exists), report per confidence bin and per document type: count and match rate; propose `REVIEW_CONFIDENCE_THRESHOLD` for a target precision among non-reviewed fields; report the MRZ detection rate per identity subtype, the writing-type distribution per document type, the review rate and flag frequencies.

---

## 15. Internal review before returning the notebook

Check your own notebook for:

* syntax errors, missing imports, undefined variables, functions called before definition, wrong paths;
* wrong Qwen class, processor/model mismatch, invalid FP8 configuration, wrong image structure, incorrect chat template, thinking not disabled;
* CPU/GPU mismatch, improper use of `device_map="auto"`, input tokens decoded as output;
* incorrect page indexing, PDFs not closed, incorrect SHA-256 behaviour, checkpoint collisions between customers or duplicate files;
* JSON / CSV serialisation problems (NumPy, Path, NaN, Arabic text, ids as strings);
* excessive generation-token limits; **a production run able to start before the gates pass**;
* `RAW_OCR_PROMPT` and generation keyword arguments byte-identical to the attached notebook;
* recorder returns `scores` unchanged, no per-step sync, no `output_scores`, argmax = generated ids, offset-map concatenation = `raw_ocr`;
* every normalised view maps back to raw offsets;
* every FR / AR marker and label string copied **verbatim** from §8;
* `DZ` / `DZA` token-bounded; biometric gate applied **before** structuring for Algerian documents; foreign rules applied only to non-Algerian documents;
* CSV read as `str` with `utf-8-sig`; `Id tiers` leading zeros preserved;
* dedup by size confirmed by SHA-256; tier plan and escalation; attempt finality; systemic breaker; review-only images; budget derived and enforced;
* Stage 2–4 runnable without the GPU; `RULES_VERSION` separate from `PIPELINE_VERSION`;
* flag enumeration enforced; conformity tables and evaluation rules exactly as §10.4 (precedence, `MISMATCH_LOW_CONFIDENCE`); CARTON "either field" rule; dossier = all processed documents; completeness separate;
* Stage 2B re-renders pages deterministically and checks `prepared_image_sha256`;
* one JSON per document per dossier + `_dossier.json`;
* no self-reported confidence; no hard-coded regulatory facts;
* the `.ipynb` JSON is structurally valid (nbformat 4) and runs cell by cell and as a Domino Job.

---

## 16. Deliverable

Your primary response must be the actual notebook:

```text
kyc_qwen36_extraction_conformite_pipeline.ipynb
```

* complete, self-contained, executable cell by cell in Domino Data Lab, valid nbformat 4 JSON;
* first Markdown cell: what the attached notebook provides, a table of every component kept / modified / added with its justification, the architecture, and how to run it;
* configuration at the top; defaults target **all dossiers**, gated by the benchmark, the regression oracle and the pre-flight;
* not code fragments, not an outline, not recommendations only.

I must be able to:

1. download it and place it in Domino next to `kyc_documents.zip` and `recertication_lissage_updated.csv`;
2. run the pre-flight on the already-validated dossier and check the regression oracle and the extracted fields;
3. launch the production run on **all** dossiers (as a Domino Job);
4. resume safely after any interruption;
5. iterate the business rules with `STAGE2_ONLY` without re-running the OCR;
6. read one JSON per document per dossier, the dossier JSON, the conformity reports and the manual-review queue.

> **Guiding principle:** keep the validated Stage 1 of the attached notebook byte-identical, capture the model's own token probabilities passively, extract every business field from the raw text with a traceable span, and let the reference file decide conformity — for every dossier in the ZIP.
