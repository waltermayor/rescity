from injector import inject

from application.bounded_contexts.analysis.domain.model._event import Event
from application.bounded_contexts.analysis.domain.model.sensor import Sensor, SensorFactory, SensorService
from application.services.dtos.event_dto import EventDto, EventDtoProducer
from application.services.dtos.sensor_create_dto import SensorCreateDto
from application.services.mappers.event_mapper import EventMapper
from application.services.dtos.sensor_created_dto import SensorCreatedDto
from application.services.mappers.sensor_created_mapper import SensorCreatedMapper
from application.services.person_tracker import PersonTracker
from application.services.dtos.person_tracked_dto import PersonTrackedDto 
from application.services.projections.sensor_current_state import SensorCurrentState, SensorCurrentStateRepository, SensorCurrentStateService 
from infrastructure.repositories.exceptions.exceptions import SensorNotFoundError
from src.application.services.person_tracker.pythonPrograms import person_tracker
from src.infrastructure.repositories.projections.sensor_current_state_inmemory import SensorCurrentStateInmemoryRepository




class SensorCurrentStateServiceV1(SensorCurrentStateService):

  @inject
  def __init__(self):
    self._sensor_current_state_repository: SensorCurrentStateRepository = SensorCurrentStateInmemoryRepository()
    
  def get_by_id(self, sensor_created_dto: SensorCreatedDto) -> SensorCreatedDto:
    sensor_id: str = sensor_created_dto.get_id()
    try:
      sensor_current_state: SensorCreatedDto = self._sensor_current_state_repository.get_by_id(sensor_id)
    except Exception as error:
      print("error capturado")
      raise error

    return sensor_current_state

  def create(self, sensor_created_dto: SensorCreatedDto) -> SensorCreatedDto:
    sensor_id: str = sensor_created_dto.get_id()
    person_tracker: PersonTracker = PersonTracker()
    sensor_current_state: SensorCurrentState = SensorCurrentState(sensor_id, person_tracker)
    self._sensor_current_state_repository.save(sensor_current_state)

    return sensor_current_state

  
  def update(self, event_dto: EventDto) -> None:
    raise NotImplementedError

  def delete(self, event_dto: EventDto) -> None:
    raise NotImplementedError

