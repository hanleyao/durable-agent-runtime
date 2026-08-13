# RAG Retrieval Evaluation Rubric

Each case is authored before retrieval and maps one query to one or more relevant document/chunk prefixes. A retrieval hit occurs when any returned `chunk_id` equals or starts with a declared relevant target.

- Recall@K: fraction of cases with at least one relevant result in the first K positions.
- MRR: mean reciprocal rank of the first relevant result.
- Category and language slices must be reported separately.
- The visible development set may be used to debug retrieval and is not eligible for final claims.
- A sealed set becomes a regression set immediately after its results are inspected or used to modify retrieval.

These metrics measure retrieval only. They do not prove final-answer correctness, faithfulness or user acceptance; those require end-to-end scoring and blinded review.
