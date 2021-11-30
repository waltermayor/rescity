from abc import ABCMeta, abstractmethod
from bson import ObjectId

class Entity(metaclass=ABCMeta):

  @abstractmethod
  def __init__(self, id: str = None) -> None:
    self._id: str = id if id != None else str(ObjectId())

  def get_id(self) -> str:
    return self._id