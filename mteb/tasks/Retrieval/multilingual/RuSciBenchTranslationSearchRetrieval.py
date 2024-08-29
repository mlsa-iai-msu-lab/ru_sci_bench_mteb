from __future__ import annotations

from mteb.abstasks import AbsTaskRetrieval, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.tasks.Retrieval.multilingual.NeuCLIR2023Retrieval import load_neuclir_data

_LANGUAGES = {
    "ru-en": ["rus-Cyrl", "eng-Latn"],
    "en-ru": ["eng-Latn", "rus-Cyrl"],
}


class RuSciBenchTranslationSearchRetrieval(MultilingualTask, AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchTranslationSearchRetrieval",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_translation_search",
            "revision": "1af517535b51163f03bdac1982728632ee78dd7c",
        },
        description="Retrieval translation of scientific papers based on their title and abstract",
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
            "n_samples": {"ru-en": 9997, "en-ru": 9969},
            "avg_character_length": {
                "ru-en": {
                    "average_document_length": 906.76,
                    "average_query_length": 871.41,
                    "num_documents": 9997,
                    "num_queries": 9969,
                    "average_relevant_docs_per_query": 1.0,
                },
                "en-ru": {
                    "average_document_length": 870.41,
                    "average_query_length": 907.76,
                    "num_documents": 9969,
                    "num_queries": 9997,
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
