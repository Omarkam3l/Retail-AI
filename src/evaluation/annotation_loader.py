import os
import json
import logging
from typing import List, Dict, Any, Optional
from src.evaluation.types import GroundTruthAnnotation, BBoxAnnotation, AnnotationFormat

logger = logging.getLogger("AnnotationLoader")


class AnnotationLoader:
    """Unified annotation loader supporting COCO, YOLO, CVAT XML, and Label Studio JSON formats."""

    def load(self, path: str, fmt: AnnotationFormat, class_names: Optional[Dict[int, str]] = None) -> List[GroundTruthAnnotation]:
        """Dispatches to the appropriate format parser."""
        if fmt == AnnotationFormat.COCO:
            return self._load_coco(path)
        elif fmt == AnnotationFormat.YOLO:
            return self._load_yolo(path, class_names or {})
        elif fmt == AnnotationFormat.CVAT:
            return self._load_cvat(path)
        elif fmt == AnnotationFormat.LABEL_STUDIO:
            return self._load_label_studio(path)
        else:
            raise ValueError(f"Unsupported annotation format: {fmt}")

    def _load_coco(self, path: str) -> List[GroundTruthAnnotation]:
        """Parses COCO JSON annotation files."""
        with open(path, "r") as f:
            data = json.load(f)

        categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        images = {img["id"]: img for img in data.get("images", [])}

        # Group annotations by image
        image_annotations: Dict[int, List[BBoxAnnotation]] = {}
        seen_ids = set()
        for ann in data.get("annotations", []):
            ann_id = ann["id"]
            if ann_id in seen_ids:
                logger.warning(f"Duplicate annotation ID {ann_id} detected, skipping.")
                continue
            seen_ids.add(ann_id)

            img_id = ann["image_id"]
            bbox = ann["bbox"]  # COCO format: [x, y, width, height]
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                logger.warning(f"Invalid bbox for annotation {ann_id}: {bbox}")
                continue

            x, y, w, h = bbox
            cat_name = categories.get(ann["category_id"], "unknown")

            box = BBoxAnnotation(
                x_min=x, y_min=y,
                x_max=x + w, y_max=y + h,
                class_label=cat_name
            )
            image_annotations.setdefault(img_id, []).append(box)

        results = []
        for img_id, img_info in images.items():
            gt = GroundTruthAnnotation(
                frame_index=img_id,
                timestamp_ms=0.0,
                image_path=img_info.get("file_name", ""),
                bboxes=image_annotations.get(img_id, [])
            )
            results.append(gt)

        logger.info(f"Loaded {len(results)} COCO annotations from {path}.")
        return results

    def _load_yolo(self, labels_dir: str, class_names: Dict[int, str]) -> List[GroundTruthAnnotation]:
        """Parses YOLO TXT label files from a directory."""
        results = []
        if not os.path.isdir(labels_dir):
            logger.warning(f"YOLO labels directory not found: {labels_dir}")
            return results

        for i, fname in enumerate(sorted(os.listdir(labels_dir))):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(labels_dir, fname)
            bboxes = []
            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # YOLO format: center_x, center_y, width, height (normalized)
                    x_min = cx - w / 2.0
                    y_min = cy - h / 2.0
                    x_max = cx + w / 2.0
                    y_max = cy + h / 2.0
                    bboxes.append(BBoxAnnotation(
                        x_min=x_min, y_min=y_min,
                        x_max=x_max, y_max=y_max,
                        class_label=class_names.get(cls_id, str(cls_id))
                    ))

            results.append(GroundTruthAnnotation(
                frame_index=i,
                timestamp_ms=0.0,
                image_path=fname.replace(".txt", ".jpg"),
                bboxes=bboxes
            ))

        logger.info(f"Loaded {len(results)} YOLO annotations from {labels_dir}.")
        return results

    def _load_cvat(self, path: str) -> List[GroundTruthAnnotation]:
        """Parses CVAT XML annotation files."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()

        results = []
        for image_elem in root.findall(".//image"):
            frame_id = int(image_elem.get("id", 0))
            img_name = image_elem.get("name", "")
            bboxes = []
            for box_elem in image_elem.findall("box"):
                label = box_elem.get("label", "unknown")
                xtl = float(box_elem.get("xtl", 0))
                ytl = float(box_elem.get("ytl", 0))
                xbr = float(box_elem.get("xbr", 0))
                ybr = float(box_elem.get("ybr", 0))
                if xbr <= xtl or ybr <= ytl:
                    logger.warning(f"Invalid CVAT bbox for frame {frame_id}")
                    continue
                bboxes.append(BBoxAnnotation(
                    x_min=xtl, y_min=ytl, x_max=xbr, y_max=ybr, class_label=label
                ))

            results.append(GroundTruthAnnotation(
                frame_index=frame_id,
                timestamp_ms=0.0,
                image_path=img_name,
                bboxes=bboxes
            ))

        logger.info(f"Loaded {len(results)} CVAT annotations from {path}.")
        return results

    def _load_label_studio(self, path: str) -> List[GroundTruthAnnotation]:
        """Parses Label Studio JSON export files."""
        with open(path, "r") as f:
            data = json.load(f)

        results = []
        for i, task in enumerate(data if isinstance(data, list) else [data]):
            bboxes = []
            for annotation in task.get("annotations", []):
                for result in annotation.get("result", []):
                    if result.get("type") != "rectanglelabels":
                        continue
                    value = result.get("value", {})
                    labels = value.get("rectanglelabels", [])
                    label = labels[0] if labels else "unknown"
                    # Label Studio uses percentage coordinates
                    x_pct = value.get("x", 0) / 100.0
                    y_pct = value.get("y", 0) / 100.0
                    w_pct = value.get("width", 0) / 100.0
                    h_pct = value.get("height", 0) / 100.0
                    bboxes.append(BBoxAnnotation(
                        x_min=x_pct, y_min=y_pct,
                        x_max=x_pct + w_pct, y_max=y_pct + h_pct,
                        class_label=label
                    ))

            img_path = task.get("data", {}).get("image", "")
            results.append(GroundTruthAnnotation(
                frame_index=i,
                timestamp_ms=0.0,
                image_path=img_path,
                bboxes=bboxes
            ))

        logger.info(f"Loaded {len(results)} Label Studio annotations from {path}.")
        return results

    def validate(self, annotations: List[GroundTruthAnnotation]) -> Dict[str, int]:
        """Validates annotations for common issues."""
        issues = {"missing_labels": 0, "invalid_bboxes": 0, "duplicate_ids": 0}
        seen_frames = set()
        for gt in annotations:
            if gt.frame_index in seen_frames:
                issues["duplicate_ids"] += 1
            seen_frames.add(gt.frame_index)
            for box in gt.bboxes:
                if not box.class_label:
                    issues["missing_labels"] += 1
                if box.x_max <= box.x_min or box.y_max <= box.y_min:
                    issues["invalid_bboxes"] += 1
        return issues
