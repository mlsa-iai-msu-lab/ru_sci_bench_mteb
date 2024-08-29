from __future__ import annotations

import logging
from typing import Any

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RegressionEvaluator import LinearRegressionEvaluator

from ..load_results.mteb_results import ScoresDict
from .AbsTask import AbsLabeledTask

logger = logging.getLogger(__name__)


class AbsTaskRegression(AbsLabeledTask):
    """Abstract class for regression tasks

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It
    must contain the following columns:
        text: str
        value: float
    """

    def __init__(self, seed: int = 42, **kwargs: Any):
        super().__init__(seed, **kwargs)
        if hasattr(self, "metadata"):
            self.metadata

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset,
        eval_split: str = "test",
        train_split: str = "train",
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]

        evaluator = LinearRegressionEvaluator(
            train_split["text"],
            train_split["value"],
            eval_split["text"],
            eval_split["value"],
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        scores = evaluator(model)
        return scores
