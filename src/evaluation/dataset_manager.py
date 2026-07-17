import os
import json
import logging
from typing import Dict, List, Optional, Any
from src.evaluation.types import DatasetMetadata, DatasetSplit

logger = logging.getLogger("DatasetManager")


class DatasetManager:
    """Version-aware dataset manager supporting train/val/test splits and multi-dataset registrations."""

    def __init__(self) -> None:
        self._datasets: Dict[str, Dict[str, Any]] = {}

    def register_dataset(self, name: str, root_path: str, version: str = "1.0") -> DatasetMetadata:
        """Registers a dataset by scanning its directory structure."""
        splits: Dict[str, int] = {}
        for split in DatasetSplit:
            split_dir = os.path.join(root_path, split.value)
            if os.path.isdir(split_dir):
                # Count image files
                count = len([f for f in os.listdir(split_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                splits[split.value] = count

        # Load metadata.json if present
        meta_path = os.path.join(root_path, "metadata.json")
        class_distribution: Dict[str, int] = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                class_distribution = meta.get("class_distribution", {})

        total_images = sum(splits.values())
        metadata = DatasetMetadata(
            name=name,
            version=version,
            num_images=total_images,
            num_annotations=0,
            class_distribution=class_distribution,
            splits=splits
        )

        self._datasets[name] = {
            "root_path": root_path,
            "version": version,
            "metadata": metadata
        }
        logger.info(f"Registered dataset '{name}' v{version} with {total_images} images across {len(splits)} splits.")
        return metadata

    def get_dataset(self, name: str) -> Optional[Dict]:
        return self._datasets.get(name)

    def list_datasets(self) -> List[str]:
        return list(self._datasets.keys())

    def get_split_path(self, name: str, split: DatasetSplit) -> Optional[str]:
        ds = self._datasets.get(name)
        if ds is None:
            return None
        return os.path.join(ds["root_path"], split.value)

    def validate_splits(self, name: str) -> Dict[str, bool]:
        """Validates that expected splits exist and are non-empty."""
        ds = self._datasets.get(name)
        if ds is None:
            return {}
        result = {}
        for split in DatasetSplit:
            path = os.path.join(ds["root_path"], split.value)
            result[split.value] = os.path.isdir(path) and len(os.listdir(path)) > 0
        return result

    def get_statistics(self, name: str) -> Optional[DatasetMetadata]:
        ds = self._datasets.get(name)
        return ds["metadata"] if ds else None
