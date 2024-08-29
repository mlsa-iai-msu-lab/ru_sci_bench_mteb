from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta

logger = logging.getLogger(__name__)


def fasttext_loader(**kwargs):
    try:
        import gensim
        import razdel
    except ImportError:
        raise ImportError(
            "gensim or razdel is not installed. Please install it with `pip install --upgrade gensim compress-fasttext razdel`."
        )

    class FastText(Encoder):
        """BM25 search"""

        def __init__(
            self,
            **kwargs,
        ):
            self._ft = gensim.models.fasttext.FastTextKeyedVectors.load(
                "/data/vatolin/ruscibench_scores/fasttext_geowac/model.model"
            )

        @classmethod
        def name(cls):
            return "fasttext_geowac"

        def _embed_text(self, text: str) -> np.ndarray[float]:
            vector = np.zeros(self._ft.vector_size)
            for token in razdel.tokenize(text):
                if any(c.isalnum() for c in token.text):
                    vector += self._ft[token.text.lower()]
            norm = sum(vector**2) ** 0.5
            if norm > 0:
                vector /= norm
            return vector

        def _embed(
            self,
            texts: list[str],
        ) -> np.ndarray:
            embedded_texts = []
            for text in texts:
                embedded_texts.append(self._embed_text(text).reshape(1, -1))
            return np.concatenate(embedded_texts)

        def encode(
            self,
            sentences: list[str],
            prompt_name: str | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            return self._embed(sentences)

        def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
            return self._embed(queries)

        def encode_corpus(self, corpus: list[dict[str, str]], **kwargs: Any) -> np.ndarray:
            texts = []
            if isinstance(corpus, dict):
                for i in range(len(corpus["text"])):  # type: ignore
                    texts.append(corpus["title"][i] + "\n" + corpus["text"][i])
            else:
                for doc in corpus:
                    texts.append(corpus["title"] + "\n" + corpus["text"])
            return self._embed(texts)

    return FastText(**kwargs)


bm25_s = ModelMeta(
    loader=fasttext_loader,  # type: ignore
    name="fasttext_geowac",
    languages=["eng_Latn"],
    open_source=True,
    revision="0_1_10",
    release_date="2024-07-10",  ## release of version 0.1.10
)
