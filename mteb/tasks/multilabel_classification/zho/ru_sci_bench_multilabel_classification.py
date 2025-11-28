from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RuSciBenchZhoMultilabelClassification(AbsTaskMultilabelClassification):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchZhoMultilabelClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_zho_multilabel",
            "revision": "e444c120de4be4d0a31103b17417651bd1b950c2",
        },
        description="",
        reference=None,
        type="MultilabelClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="f1",
        date=None,
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators="derived",
        dialect=None,
        sample_creation=None,
        prompt="Classify the category of scientific papers based on the titles and abstracts",
        bibtex_citation=""
    )
