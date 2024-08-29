from __future__ import annotations

from mteb.abstasks import AbsTaskClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "grnti_ru": ["rus-Cyrl"],
    "grnti_en": ["eng-Latn"],
}


class RuSciBenchGRNTIOrigClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchGRNTIOrigClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="Classification of scientific papers (title+abstract) by rubric",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Topic classification"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 2640},
            "avg_character_length": {"train": 891.13, "test": 891.23},
        },
    )
