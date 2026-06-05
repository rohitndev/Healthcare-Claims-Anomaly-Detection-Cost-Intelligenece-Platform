"""ClinicalBERT-based diagnosis-procedure alignment scorer.

Validates whether a claim's free-text discharge note clinically supports the
billed ICD-10 diagnosis. When ``transformers``/``torch`` and the ClinicalBERT
weights are available, the note and a diagnosis prompt are embedded with
ClinicalBERT (``emilyalsentzer/Bio_ClinicalBERT``) and compared by cosine
similarity. Otherwise a deterministic lexical aligner scores the overlap between
the note and the diagnosis's expected clinical keywords. Both return a
``misalignment`` score in 0-1 where higher means the note does not match the
diagnosis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.taxonomy.taxonomy_graph import TaxonomyGraph
from src.utils.logger import get_logger

logger = get_logger("clinical-bert")


class ClinicalBERTAligner:
    def __init__(self, graph: TaxonomyGraph | None = None) -> None:
        self.graph = graph or TaxonomyGraph.from_reference()
        self._model = None
        self._tokenizer = None
        self.backend = "lexical"
        if CONFIG.nlp.use_transformer:
            self._try_load_model()

    def _try_load_model(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(CONFIG.nlp.model_name)
            self._model = AutoModel.from_pretrained(CONFIG.nlp.model_name)
            self._model.eval()
            self.backend = "clinicalbert"
            logger.info("Loaded ClinicalBERT model: %s", CONFIG.nlp.model_name)
        except Exception as exc:
            logger.warning("ClinicalBERT unavailable, using lexical aligner (%s)", exc)
            self.backend = "lexical"

    # ------------------------------------------------------------------ #
    def _embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - heavy path
        import torch

        with torch.no_grad():
            enc = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=CONFIG.nlp.max_length, return_tensors="pt",
            )
            out = self._model(**enc)
            # Mean-pool the last hidden state.
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            return (summed / counts).cpu().numpy()

    def _lexical_misalignment(self, note: str, icd10: str) -> float:
        keywords = self.graph.note_keywords(icd10)
        if not keywords:
            return 0.0
        note_l = (note or "").lower()
        hits = sum(1 for kw in keywords if kw in note_l)
        coverage = hits / len(keywords)
        return float(np.clip(1.0 - coverage, 0.0, 1.0))

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        notes = out["discharge_note"].fillna("").tolist()
        icds = out["icd10_code"].tolist()

        if self.backend == "clinicalbert" and self._model is not None:  # pragma: no cover
            prompts = [f"Clinical presentation of {self.graph.describe(i)}" for i in icds]
            note_emb = self._embed(notes)
            dx_emb = self._embed(prompts)
            sims = (note_emb * dx_emb).sum(1) / (
                np.linalg.norm(note_emb, axis=1) * np.linalg.norm(dx_emb, axis=1) + 1e-9
            )
            sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
            out["clinical_misalignment"] = (1.0 - sims).clip(0, 1)
        else:
            out["clinical_misalignment"] = [
                self._lexical_misalignment(n, i) for n, i in zip(notes, icds)
            ]

        out["note_alignment_backend"] = self.backend
        logger.info(
            "Scored diagnosis-procedure alignment for %d notes (backend=%s, mean misalignment=%.3f)",
            len(out), self.backend, float(out["clinical_misalignment"].mean()),
        )
        return out
