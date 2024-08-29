from __future__ import annotations

from mteb.abstasks import AbsTaskRetrieval, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.tasks.Retrieval.multilingual.NeuCLIR2023Retrieval import load_neuclir_data

_LANGUAGES = {
    "ru": ["rus-Cyrl"],
    "en": ["eng-Latn"],
}


class RuSciBenchCociteRetrieval(MultilingualTask, AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCociteRetrieval",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval",
            "revision": "a5da47a245275669d2b6ddf8f96c5338dd2428b4",
        },
        description="Retrieval of related scientific papers based on their title and abstract",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="MIT",
        dialect=[],
        sample_creation="found",
        annotations_creators="derived",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"ru": 90000, "en": 90000},
            "avg_character_length": {
                "ru": {
                    "average_document_length": 891.66,
                    "average_query_length": 990.93,
                    "num_documents": 90000,
                    "num_queries": 3000,
                    "average_relevant_docs_per_query": 1.0,
                },
                "en": {
                    "average_document_length": 930.81,
                    "average_query_length": 1023.72,
                    "num_documents": 90000,
                    "num_queries": 3000,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_neuclir_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
