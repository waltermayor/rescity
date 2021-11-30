from abc import ABCMeta, abstractmethod
from datetime import datetime
from bson import ObjectId

class Event(metaclass=ABCMeta):

  @abstractmethod
  def __init__(self, type: str, aggregate_id: str, aggregate_type: str, data: dict, id: str = None, created_datetime: datetime = None) -> None:
    self._id: str = id if id != None else str(ObjectId())
    self._type: str = type
    self._aggregate_id: str = aggregate_id
    self._aggregate_type: str = aggregate_type
    self._data: dict = data
    self._created_datetime: datetime = created_datetime if created_datetime != None else datetime.utcnow()

  def get_id(self) -> str:
    return self._id

  def get_type(self) -> str:
    return self._type

  def get_aggregate_id(self) -> str:
    return self._aggregate_id

  def get_aggregate_type(self) -> str:
    return self._aggregate_type

  def get_data(self) -> dict:
    return self._data

  def get_created_datetime(self) -> datetime:
    return self._created_datetime
