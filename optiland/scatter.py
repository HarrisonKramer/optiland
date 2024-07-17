from abc import ABC, abstractmethod


class BaseBSDF(ABC):

    @abstractmethod
    def scatter(self):
        pass
