import json
import os
import logging
from typing import List, Dict, Optional
from src.evaluation.types import GroundTruthAnnotation, BBoxAnnotation

logger = logging.getLogger("DatasetConverters")


class DatasetConverter:
    """Bidirectional converters between annotation formats: COCO, YOLO, CVAT, Label Studio."""

    def coco_to_yolo(self, annotations: List[GroundTruthAnnotation],
                     class_to_id: Dict[str, int],
                     output_dir: str, img_width: int = 1, img_height: int = 1) -> int:
        """Converts COCO-style annotations to YOLO TXT format files."""
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for gt in annotations:
            fname = os.path.splitext(os.path.basename(gt.image_path))[0] + ".txt"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w") as f:
                for box in gt.bboxes:
                    cls_id = class_to_id.get(box.class_label, 0)
                    cx = ((box.x_min + box.x_max) / 2.0) / img_width
                    cy = ((box.y_min + box.y_max) / 2.0) / img_height
                    w = (box.x_max - box.x_min) / img_width
                    h = (box.y_max - box.y_min) / img_height
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    count += 1
        logger.info(f"Converted {count} annotations to YOLO format in {output_dir}.")
        return count

    def yolo_to_coco(self, annotations: List[GroundTruthAnnotation],
                     id_to_class: Dict[int, str]) -> Dict:
        """Converts YOLO-style annotations to COCO JSON structure."""
        categories = [{"id": cid, "name": cname} for cid, cname in id_to_class.items()]
        images = []
        coco_annotations = []
        ann_id = 1
        for gt in annotations:
            img_id = gt.frame_index
            images.append({"id": img_id, "file_name": gt.image_path})
            for box in gt.bboxes:
                w = box.x_max - box.x_min
                h = box.y_max - box.y_min
                cat_id = next((cid for cid, cn in id_to_class.items() if cn == box.class_label), 0)
                coco_annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_id,
                    "bbox": [box.x_min, box.y_min, w, h], "area": w * h, "iscrowd": 0
                })
                ann_id += 1

        return {"images": images, "annotations": coco_annotations, "categories": categories}

    def coco_to_cvat(self, annotations: List[GroundTruthAnnotation]) -> str:
        """Converts COCO-style annotations to CVAT XML string."""
        import xml.etree.ElementTree as ET
        root = ET.Element("annotations")
        for gt in annotations:
            img_elem = ET.SubElement(root, "image", id=str(gt.frame_index), name=gt.image_path)
            for box in gt.bboxes:
                ET.SubElement(img_elem, "box", label=box.class_label,
                              xtl=str(box.x_min), ytl=str(box.y_min),
                              xbr=str(box.x_max), ybr=str(box.y_max))
        return ET.tostring(root, encoding="unicode")

    def label_studio_to_coco(self, annotations: List[GroundTruthAnnotation],
                             class_to_id: Dict[str, int]) -> Dict:
        """Converts Label Studio annotations to COCO JSON structure."""
        return self.yolo_to_coco(annotations, {v: k for k, v in class_to_id.items()})
