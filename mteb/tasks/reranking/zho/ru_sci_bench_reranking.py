from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RuSciBenchReranking(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchReranking",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_zho_cite_reranking",
            "revision": "ae23f294e61749ba1ecdfd42186e7c51226344ab",
        },
        description="",
        reference=None,
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=[],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        prompt={"query": "Given a title and abstract of a scientific paper, rerank the titles and abstracts of other relevant papers"},
        bibtex_citation=""
)

