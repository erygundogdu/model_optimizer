from abc import ABC, abstractmethod

from optimizer.core.model_bundle import ModelBundle


class ModelAdapter(ABC):

    @abstractmethod
    def load(
        self,
        weights_path: str,
        device: str
    ) -> ModelBundle:
        pass

    @abstractmethod
    def save(
        self,
        bundle: ModelBundle,
        output_path: str
    ) -> None:
        pass