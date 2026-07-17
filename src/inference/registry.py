from typing import Dict, Type, Any
from src.common.interfaces import Lifecycle
from src.inference.interfaces import BaseInferencePipeline

class ModelRegistry:
    """Registry class managing the paths and states of compiled model weights (TensorRT/ONNX)."""
    
    _models: Dict[str, str] = {}

    @classmethod
    def register_model(cls, model_name: str, model_path: str) -> None:
        """Register a path for a given model weight key."""
        cls._models[model_name] = model_path

    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Retrieve the registered file path of a model."""
        if model_name not in cls._models:
            raise KeyError(f"Model '{model_name}' is not registered in the Model Registry.")
        return cls._models[model_name]


class PipelineRegistry:
    """Registry class managing the instantiation of processing pipelines."""
    
    _pipelines: Dict[str, Type[BaseInferencePipeline]] = {}

    @classmethod
    def register_pipeline(cls, name: str, pipeline_class: Type[BaseInferencePipeline]) -> None:
        """Register a pipeline class type."""
        cls._pipelines[name] = pipeline_class

    @classmethod
    def get_pipeline(cls, name: str) -> Type[BaseInferencePipeline]:
        """Retrieve a registered pipeline class type."""
        if name not in cls._pipelines:
            raise KeyError(f"Pipeline '{name}' is not registered in the Pipeline Registry.")
        return cls._pipelines[name]
