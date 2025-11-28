from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RuSciBenchZhoClassification(AbsTaskClassification):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchZhoClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_zho_multiclass",
            "revision": "5789f0b78d6ea7bbaf8f97a37140a1c109b27851",
        },
        description="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        reference=None,
        type="Classification",
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
        dialect=[],
        sample_creation=None,
        prompt="Classify the category of scientific papers based on the titles and abstracts",
        bibtex_citation=""
)
