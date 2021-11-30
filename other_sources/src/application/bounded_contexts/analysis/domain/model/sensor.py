from abc import ABCMeta, abstractmethod
from typing import Tuple, List

from application.bounded_contexts.analysis.domain.model._entity import Entity
from application.bounded_contexts.analysis.domain.model._event import Event

from application.services.dtos.event_dto import EventDto
from application.services.dtos.sensor_create_dto import SensorCreateDto
from application.services.dtos.sensor_created_dto import SensorCreatedDto
from src.application.services.person_tracker.pythonPrograms.person_tracker import PersonTracker
from src.application.services.projections.sensor_current_state import SensorCurrentState

class Sensor(Entity):

  def __init__(self, id: str = None) -> None:
    super().__init__(id)

  # Events

  class Created(Event):
    def __init__(self, sensor_id: str, img: str, created_datetime: str) -> None:
      data: dict = {
        'img': img,
        'created_datetime': created_datetime
      }
      super().__init__('created', sensor_id, 'sensor', data)

  class Calculated(Event):
    def __init__(self, sensor_id: str, place_id:str, number_of_persons: int, number_of_close_persons: int, persons_without_mask: int) -> None:
      data: dict = {
          "place_id": place_id,
          "measurements":[
          {
            "name": "occupation",
            "value": number_of_persons
          },
          {
            "name": "distance",
            "value": number_of_close_persons
          },
          {
            "name": "persons_without_mask",
            "value": persons_without_mask
          }
        ]
      }
      super().__init__('calculated', sensor_id, 'sensor', data)

  # Behaviours

  # TODO: Type constant list
  def calculate(self, sensor_current_state: SensorCurrentState, place_id: str, img: str) -> Event:
    sensor_id: str = sensor_current_state.get_id()
    person_tracker: PersonTracker = sensor_current_state.get_person_tracker()
    number_of_close_persons, number_of_persons, persons_without_mask =person_tracker.analyseImage(img)

    print("pesrons", number_of_close_persons, number_of_persons, persons_without_mask)
    return Sensor.Calculated(sensor_id, place_id, number_of_persons,number_of_close_persons, persons_without_mask)
    #return Sensor.Calculated(sensor_id, place_id,2,4,5)


class SensorFactory():

  def create(img: str, created_datetime: int) -> Tuple[Sensor, Event]:
    sensor: Sensor = Sensor()
    event: Event = Sensor.Created(sensor.get_id(), img, created_datetime)
    return sensor, event

  def instantiate(id: str) -> Sensor:
    return Sensor(id)


class SensorService(metaclass=ABCMeta):

  @abstractmethod
  def create(self, sensor_create_dto: SensorCreateDto) -> SensorCreatedDto:
    raise NotImplementedError

  @abstractmethod
  def calculate(self, sensor_create_dto: SensorCreateDto) -> None:
    raise NotImplementedError
