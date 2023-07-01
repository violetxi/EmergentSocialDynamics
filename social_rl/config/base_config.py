from typeguard import typechecked
from abc import ABC, abstractmethod


@typechecked
class BaseConfig(metaclass=ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass