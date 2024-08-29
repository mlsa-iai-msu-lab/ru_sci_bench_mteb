from mteb.abstasks import AbsTaskRegression, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "cited_count_ru": ["rus-Cyrl"],
    "cited_count_en": ["eng-Latn"],
}


class RuSciBenchCitedCountRegression(MultilingualTask, AbsTaskRegression):
    metadata = TaskMetadata(
        name="RuSciBenchCitedCountRegression",
        description="Regression on scientific papers (title+abstract) to predict the number of citations",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        type="Regression",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="kendalltau",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="MIT",
        sample_creation="found",
        annotations_creators="derived",
        dialect=None,
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 18227},
            "avg_character_length": {"train": 893.74, "test": 905.17},
        },
    )

    def dataset_transform(self):
        for subset in self.dataset:
            self.dataset[subset]["train"] = self.dataset[subset][
                "train"
            ].train_test_split(test_size=2048, seed=self.seed)["test"]
