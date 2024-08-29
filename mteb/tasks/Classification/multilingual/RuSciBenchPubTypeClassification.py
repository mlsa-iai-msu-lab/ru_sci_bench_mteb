from __future__ import annotations

from mteb.abstasks import AbsTaskClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "pub_type_ru": ["rus-Cyrl"],
    "pub_type_en": ["eng-Latn"],
}


class RuSciBenchPubTypeClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchPubTypeClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="Classification of scientific papers (title+abstract) by publication type",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 462},
            "avg_character_length": {"train": 756.72, "test": 757.81},
        },
    )
