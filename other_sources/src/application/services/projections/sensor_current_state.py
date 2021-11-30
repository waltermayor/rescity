from abc import ABCMeta, abstractmethod
from typing import List

from application.services.projections._projection import Projection
from application.services.dtos.event_dto import EventDto
from application.services.dtos.sensor_created_dto import SensorCreatedDto
from application.services.person_tracker import PersonTracker

class SensorCurrentState(Projection):

  def __init__(self, id: str, person_tracker: PersonTracker, created_datetime:str = None, updated_datetime:str = None) -> None:
    super().__init__(id, created_datetime, updated_datetime)
    self._person_tracker: PersonTracker  = person_tracker

  def get_person_tracker(self) -> str:
    return self._person_tracker

  def set_person_tracker(self, person_tracker: str) -> None:
    self._person_tracker = person_tracker
    self.set_updated_datetime()



class SensorCurrentStateService(metaclass=ABCMeta):

  @abstractmethod
  def get_by_id(self, sensor_created_dto: SensorCreatedDto) -> SensorCurrentState:
    raise NotImplementedError

  @abstractmethod
  def create(self, sensor_created_dto: SensorCreatedDto) -> SensorCurrentState:
    raise NotImplementedError

  @abstractmethod
  def update(self, event_dto: EventDto) -> None:
    raise NotImplementedError

  @abstractmethod
  def delete(self, event_dto: EventDto) -> None:
    raise NotImplementedError
 

class SensorCurrentStateRepository(metaclass=ABCMeta):

  @abstractmethod
  def get_by_id(self, id: str) -> SensorCurrentState:
    raise NotImplementedError

  @abstractmethod
  def get_all(self) -> List[SensorCurrentState]:
    raise NotImplementedError

  @abstractmethod
  def save(self, sensor_current_state: SensorCurrentState) -> None:
    raise NotImplementedError

  @abstractmethod
  def delete(self, sensor_current_state: SensorCurrentState) -> None:
    raise NotImplementedError