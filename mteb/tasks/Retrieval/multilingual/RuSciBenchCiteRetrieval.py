from __future__ import annotations

from mteb.abstasks import AbsTaskRetrieval, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.tasks.Retrieval.multilingual.NeuCLIR2023Retrieval import load_neuclir_data

_LANGUAGES = {
    "ru": ["rus-Cyrl"],
    "en": ["eng-Latn"],
}


class RuSciBenchCiteRetrieval(MultilingualTask, AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCiteRetrieval",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval",
            "revision": "6cb447d02f41b8b775d5d9df7faf472f44d2f1db",
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
                    "average_document_length": 899.09,
                    "average_query_length": 1419.35,
                    "num_documents": 90000,
                    "num_queries": 3000,
                    "average_relevant_docs_per_query": 1.0,
                },
                "en": {
                    "average_document_length": 939.21,
                    "average_query_length": 1486.08,
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
