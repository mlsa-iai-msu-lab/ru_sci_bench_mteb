from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class RuSciBenchOECDClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchOECDClassification",
        dataset={
            "path": "ai-forever/ru-scibench-oecd-classification",
            "revision": "26c88e99dcaba32bb45d0e1bfc21902337f6d471",
        },
        description="Classification of scientific papers (title+abstract) by rubric",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench/",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
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
            "n_samples": {"test": 2048},
            "avg_character_length": {"test": 838.9},
        },
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )
