from abc import ABCMeta, abstractmethod
from datetime import datetime

class Projection(metaclass=ABCMeta):

  @abstractmethod
  def __init__(self, id: str, created_datetime:datetime = None, updated_datetime:datetime = None) -> None:
    self._id: str = id
    self._created_datetime: datetime = created_datetime if created_datetime != None else datetime.utcnow()
    self._updated_datetime: datetime = updated_datetime if created_datetime != None else datetime.utcnow()

  def get_id(self) -> str:
    return self._id

  def get_created_datetime(self) -> datetime:
    return self._created_datetime

  def get_updated_datetime(self) -> datetime:
    return self._updated_datetime

  def set_updated_datetime(self) -> datetime:
    self._updated_datetime = datetime.utcnow()