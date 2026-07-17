import json
import os
import logging
import uuid
from typing import List, Optional, Dict, Any
from src.evaluation.types import ExperimentRecord, OverallEvaluationResult

logger = logging.getLogger("ExperimentTracker")

class ExperimentTracker:
    """Stores, loads, and compares experiment records in structured JSON."""

    def __init__(self, storage_dir: str = "experiments") -> None:
        self._storage_dir = storage_dir
        self._experiments: Dict[str, ExperimentRecord] = {}
        os.makedirs(storage_dir, exist_ok=True)

    def create_experiment(
        self,
        git_commit: str,
        dataset_version: str,
        model_version: str,
        configuration: Dict[str, Any] = None,
        thresholds: Dict[str, float] = None,
        notes: str = ""
    ) -> ExperimentRecord:
        """Creates a new experiment record with a unique ID."""
        exp_id = str(uuid.uuid4())[:8]
        record = ExperimentRecord(
            experiment_id=exp_id,
            git_commit=git_commit,
            dataset_version=dataset_version,
            model_version=model_version,
            configuration=configuration or {},
            thresholds=thresholds or {},
            notes=notes
        )
        self._experiments[exp_id] = record
        logger.info(f"Created experiment '{exp_id}'.")
        return record

    def record_metrics(self, experiment_id: str, metrics: OverallEvaluationResult,
                       execution_time: float = 0.0) -> None:
        """Associates evaluation metrics with an experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found.")
        exp = self._experiments[experiment_id]
        exp.metrics = metrics
        exp.execution_time_seconds = execution_time

    def save(self, experiment_id: str) -> str:
        """Saves an experiment record to disk as JSON."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found.")
        exp = self._experiments[experiment_id]
        fpath = os.path.join(self._storage_dir, f"{experiment_id}.json")

        data = {
            "experiment_id": exp.experiment_id,
            "git_commit": exp.git_commit,
            "dataset_version": exp.dataset_version,
            "model_version": exp.model_version,
            "configuration": exp.configuration,
            "thresholds": exp.thresholds,
            "execution_time_seconds": exp.execution_time_seconds,
            "notes": exp.notes,
            "metrics_available": exp.metrics is not None
        }

        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved experiment '{experiment_id}' to {fpath}.")
        return fpath

    def load(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Loads an experiment from disk or cache."""
        if experiment_id in self._experiments:
            return self._experiments[experiment_id]

        fpath = os.path.join(self._storage_dir, f"{experiment_id}.json")
        if not os.path.isfile(fpath):
            return None

        with open(fpath, "r") as f:
            data = json.load(f)

        record = ExperimentRecord(
            experiment_id=data["experiment_id"],
            git_commit=data["git_commit"],
            dataset_version=data["dataset_version"],
            model_version=data["model_version"],
            configuration=data.get("configuration", {}),
            thresholds=data.get("thresholds", {}),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            notes=data.get("notes", "")
        )
        self._experiments[experiment_id] = record
        return record

    def compare(self, experiment_ids: List[str]) -> List[Dict[str, Any]]:
        """Compares multiple experiments side-by-side."""
        results = []
        for eid in experiment_ids:
            exp = self._experiments.get(eid)
            if exp is None:
                continue
            entry = {
                "id": exp.experiment_id,
                "model": exp.model_version,
                "dataset": exp.dataset_version,
                "time_s": f"{exp.execution_time_seconds:.2f}",
                "has_metrics": exp.metrics is not None
            }
            results.append(entry)
        return results

    def list_experiments(self) -> List[str]:
        return list(self._experiments.keys())
