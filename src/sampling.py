import io
from typing import Annotated, Any, Dict, Iterator, List, Tuple

import numpy as np
from pathlib import Path

from datamaestro_text.data.ir import (
    Adhoc,
    PairwiseSampleDataset,
    TextItem,
    IDItem,
)
from experimaestro import Param, tqdm, Task, Annotated, pathgenerator, Constant
from xpmir.rankers import ScoredDocument, Retriever
from xpmir.learning import Sampler
from xpmir.utils.functools import cache as cache
from experimaestro.annotations import cache as xpmir_cache
from xpmir.letor.samplers import TSVPairwiseSampleDataset, PairwiseModelBasedSampler
from xpmir.letor.records import PointwiseRecord
from xpmir.utils.iter import SerializableIterator
from xpmir.papers.helpers.samplers import ValidationSample, prepare_collection, RandomFold

import logging

logging.basicConfig(level=logging.INFO)    

@cache
def generic_dataset(cfg: ValidationSample, dataset_name: str, exclude_dataset_name: str = None, launcher=None):
    """Sample dev topics to get a validation subset"""
    return RandomFold.C(
        dataset=prepare_collection(dataset_name),
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
        exclude=prepare_collection(exclude_dataset_name).topics if exclude_dataset_name is not None else None,
    ).submit(launcher=launcher)

class TopPassagesSampler(Task, Sampler):
    """Used to get the top k passages per query.
    
    Output is written to a tsv file.
    """
    __xpmid__="src.ablations.ablation_utils.toppassagessampler"

    dataset: Param[Adhoc]
    """The IR adhoc dataset"""

    retriever: Param[Retriever]
    """A retriever to sample negative documents"""

    dataset_path: Annotated[Path, pathgenerator("dataset.tsv")]

    version: Constant[int] = 1

    def task_outputs(self, dep) -> PairwiseSampleDataset:
        """return a iterator of PairwiseSample"""
        return dep(
            TSVPairwiseSampleDataset.C(
                id=self.dataset.id,
                hard_negative_samples_path=self.dataset_path,
            )
        )

    def execute(self):
        """Retrieve over the dataset and store the passages
        """
        self.logger.info("Reading topics and retrieving documents")

        tmprunpath = self.dataset_path.with_suffix(".tmp")

        with tmprunpath.open("wt") as fp:

            # Read the assessments
            self.logger.info("Reading assessments")
            assessments: Dict[str, Dict[str, float]] = {} 
            for qrels in self.dataset.assessments.iter():
                doc2rel = {}
                assessments[qrels.topic_id] = doc2rel
                for qrel in qrels.assessments:
                    doc2rel[qrel.doc_id] = qrel.rel

            self.logger.info("Assessment loaded")
            self.logger.info("Read assessments for %d topics", len(assessments))

            self.logger.info("Retrieving documents for each topic")
            queries = []
            for query in self.dataset.topics.iter():
                queries.append(query)

            # count the number of queries been skipped because of no assessments
            # available
            skipped = 0
            for query in tqdm(queries):
                q_fp = io.StringIO()
                qassessments = assessments.get(query[IDItem].id, None)
                if not qassessments:
                    skipped += 1
                    self.logger.warning("Skipping topic %s (no assessments)", query[IDItem].id)
                    continue
                
                scoreddocuments: List[ScoredDocument] = self.retriever.retrieve(query)
                passages = []
                for sd in scoreddocuments:
                    passages.append(sd.document[IDItem].id)

                if not passages:
                    self.logger.debug(
                        "Skipping topic %s (no documents)", query[IDItem].id
                    )
                    skipped += 1
                    continue

                assert len(passages) > 0
                # Write the documents
                passages_str = " ".join(passages)
                q_fp.write(
                    f"{query[IDItem].id}\tpositives:\t{passages_str}\tnegatives:\t" "\n"
                )
                fp.write(q_fp.getvalue())
                q_fp.close()

        self.logger.info("Processed %d topics (%d skipped)", len(queries), skipped)
        self.logger.info("Dataset written to %s", self.dataset_path)
        tmprunpath.rename(self.dataset_path)

class DiversePassagesSamplerWithHardNegatives(Task, Sampler):
    """Retriever-based hard negative sampler but conditionned on the maximum
    number of documents to sample per query IN TOTAL AND PER RELEVANCE LEVEL.
    
    In practice, for each query, we start by reading the assessments already existing but filter them.
    Filter takes into account a maximum number of documents per relevance level.
    
    If after the first filtering, we have less than the maximum number of documents per relevance level, we
    sample the remaining documents from the retriever."""

    __xpmid__="src.evd.utils.diversepassagessamplerwithhardnegatives"

    dataset: Param[Adhoc]
    """The IR adhoc dataset"""

    retriever: Param[Retriever]
    """A retriever to sample negative documents"""

    relevance_levels: Param[List[int]]
    """List of relevance levels to consider as relevant for the query"""

    max_passages_per_relevance_level: Param[int] = 20
    """Maximum number of passages to sample per relevance level"""

    max_passages_per_query: Param[int] = 100
    """Maximum number of passages to sample per query"""

    dataset_path: Annotated[Path, pathgenerator("dataset.tsv")]

    def task_outputs(self, dep) -> PairwiseSampleDataset:
        """return a iterator of PairwiseSample"""
        return dep(
            TSVPairwiseSampleDataset.C(
                id=self.dataset.id,
                hard_negative_samples_path=self.dataset_path,
            )
        )

    def execute(self):
        """Retrieve over the dataset and select the positive and negative
        according to the relevance score and their rank
        """
        self.logger.info("Reading topics and retrieving documents")

        tmprunpath = self.dataset_path.with_suffix(".tmp")

        with tmprunpath.open("wt") as fp:

            # Read the assessments
            self.logger.info("Reading assessments")
            assessments: Dict[str, Dict[str, float]] = {} 
            less_than_max_passages_per_query_list = list() # list of query_id where with less than max_passages_per_query assessments
            for qrels in self.dataset.assessments.iter():
                doc2rel = {}
                assessments[qrels.topic_id] = doc2rel
                relevance_levels_counters = dict() 
                assessment_count = 0               
                for qrel in qrels.assessments:
                    if qrel.rel < 0:
                        # If the relevance level is negative, we regard it as weak_negative and skip it
                        continue
                    if qrel.rel in relevance_levels_counters.keys() and relevance_levels_counters[qrel.rel] >= self.max_passages_per_relevance_level:
                        # If that's the case, we move onto the next assessment
                        continue
                    else:
                        doc2rel[qrel.doc_id] = qrel.rel
                        relevance_levels_counters[qrel.rel] = relevance_levels_counters.get(qrel.rel, 0) + 1
                        assessment_count += 1
                
                if assessment_count < self.max_passages_per_query:
                    # If we have less than max_passages_per_query assessments, we add the query_id to the list
                    less_than_max_passages_per_query_list.append(qrels.topic_id)

            self.logger.info("Assessment loaded")
            self.logger.info("Read assessments for %d topics", len(assessments))

            self.logger.info("Retrieving documents for each topic")
            queries = []
            for query in self.dataset.topics.iter():
                queries.append(query)

            # count the number of queries been skipped because of no assessments
            # available
            skipped = 0
            for query in tqdm(queries):
                q_fp = io.StringIO()
                qassessments = assessments.get(query[IDItem].id, None)
                if not qassessments:
                    skipped += 1
                    self.logger.warning("Skipping topic %s (no assessments)", query[IDItem].id)
                    continue

                # Write all the positive documents
                positives = []
                negatives = []
                # First we store all the documents we parsed from the assessments
                for docno, rel in qassessments.items():
                    if rel in self.relevance_levels:
                        # It is a positive document
                        positives.append(docno)
                
                if not positives:
                    self.logger.warning(
                        "Skipping topic %s (no relevant documents)",
                        query[IDItem].id,
                    )
                    skipped += 1
                    continue

                for docno, rel in qassessments.items():
                    if rel not in self.relevance_levels:
                        negatives.append(docno)
                
                # Second, in the case where we are missing documents, we sample hard negatives to fill the gap
                if query[IDItem].id in less_than_max_passages_per_query_list:
                    scoreddocuments: List[ScoredDocument] = self.retriever.retrieve(query)

                    missing_docs_count = self.max_passages_per_query - len(positives) - len(negatives)
                    for rank, sd in enumerate(scoreddocuments):
                        rel = qassessments.get(sd.document[IDItem].id, 0)
                        if rel > 0:
                            continue
                        else:
                            # It is a negative document or
                            # don't exist in assessment
                            negatives.append(sd.document[IDItem].id)
                            missing_docs_count -= 1
                            if missing_docs_count <= 0:
                                break

                if not negatives:
                    self.logger.debug(
                        "Skipping topic %s (no negative documents)", query[IDItem].id
                    )
                    skipped += 1
                    continue

                assert len(positives) > 0 and len(negatives) > 0
                # Write the positive and negative documents
                positive_str = " ".join(positives)
                negative_str = " ".join(negatives)
                q_fp.write(
                    f"{query[IDItem].id}\tpositives:\t{positive_str}\t"
                    f"negatives:\t{negative_str}\n"
                )
                fp.write(q_fp.getvalue())
                q_fp.close()

        self.logger.info("Processed %d topics (%d skipped)", len(queries), skipped)
        self.logger.info("Dataset written to %s", self.dataset_path)
        tmprunpath.rename(self.dataset_path)


class AttributionInfoNCESampler(PairwiseModelBasedSampler):
    """A pairwise sampler based on a retrieval model"""

    __xpmid__="src.attribution_methods.samplers.attributioninfoncesampler"

    relevant_levels: Param[List[int]]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    @xpmir_cache("run")
    def _itertopics(
        self, runpath: Path
    ) -> Iterator[
        Tuple[str, List[Tuple[str, int, float]], List[Tuple[str, int, float]]]
    ]:
        """Iterates over topics, returning retrieved positives and negatives
        documents"""
        self.logger.info("Reading topics and retrieving documents")

        if not runpath.is_file():
            tmprunpath = runpath.with_suffix(".tmp")

            with tmprunpath.open("wt") as fp:

                # Read the assessments
                self.logger.info("Reading assessments")
                assessments: Dict[str, Dict[str, float]] = {}
                for qrels in self.dataset.assessments.iter():
                    doc2rel = {}
                    assessments[qrels.topic_id] = doc2rel
                    for qrel in qrels.assessments:
                        doc2rel[qrel.doc_id] = qrel.rel
                self.logger.info("Read assessments for %d topics", len(assessments))

                self.logger.info("Retrieving documents for each topic")
                queries = []
                for query in self.dataset.topics.iter():
                    queries.append(query)

                # Retrieve documents
                skipped = 0
                for query in tqdm(queries):
                    q_fp = io.StringIO()
                    qassessments = assessments.get(query[IDItem].id, None)
                    if not qassessments:
                        skipped += 1
                        self.logger.warning(
                            "Skipping topic %s (no assessments)", query[IDItem].id
                        )
                        continue

                    # Write all the positive documents
                    positives = []
                    for docno, rel in qassessments.items():
                        if rel in self.relevant_levels:
                            q_fp.write(
                                f"{query[TextItem].text if not positives else ''}"
                                f"\t{docno}\t0.\t{rel}\n"
                            )
                            positives.append((docno, rel, 0))

                    if not positives:
                        self.logger.warning(
                            "Skipping topic %s (no relevant documents)",
                            query[IDItem].id,
                        )
                        skipped += 1
                        continue

                    scoreddocuments: List[ScoredDocument] = self.retriever.retrieve(
                        query
                    )

                    negatives = []
                    for rank, sd in enumerate(scoreddocuments):
                        # Get the assessment (assumes not relevant)
                        rel = qassessments.get(sd.document[IDItem].id, 0)
                        if rel in self.relevant_levels:
                            continue

                        negatives.append((sd.document[IDItem].id, rel, sd.score))
                        q_fp.write(f"\t{sd.document[IDItem].id}\t{sd.score}\t{rel}\n")

                    if not negatives:
                        self.logger.warning(
                            "Skipping topic %s (no negatives documents)",
                            query[IDItem].id,
                        )
                        skipped += 1
                        continue

                    assert len(positives) > 0 and len(negatives) > 0

                    # Write in cache, and yield
                    fp.write(q_fp.getvalue())
                    q_fp.close()
                    yield query[TextItem].text, positives, negatives

                # Finally, move the cache file in place...
                self.logger.info(
                    "Processed %d topics (%d skipped)", len(queries), skipped
                )
                tmprunpath.rename(runpath)
        else:
            # Read from cache
            self.logger.info("Reading records from file %s", runpath)
            with runpath.open("rt") as fp:
                positives = []
                negatives = []
                oldtitle = ""

                for line in fp.readlines():
                    title, docno, score, rel = line.rstrip().split("\t")
                    if title:
                        if oldtitle:
                            yield oldtitle, positives, negatives
                        positives = []
                        negatives = []
                    else:
                        title = oldtitle
                    title = title or oldtitle
                    rel = int(rel)
                    (positives if rel in self.relevant_levels else negatives).append(
                        (docno, rel, float(score))
                    )
                    oldtitle = title

                yield oldtitle, positives, negatives

    def sampler_iter(self) -> SerializableIterator[PointwiseRecord, Any]:
        """Iterable over pointwise records"""
        return self.pairwise_iter()