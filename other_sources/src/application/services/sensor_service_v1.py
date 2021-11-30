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
from application.services.dtos.sensor_publish_data_dto import SensorPublishDataDto
from src.application.services import person_tracker
from src.application.services.projections.sensor_current_state import SensorCurrentStateService, SensorCurrentState 
from infrastructure.repositories.exceptions.exceptions import SensorNotFoundError



class SensorServiceV1(SensorService):

  @inject
  def __init__(self, sensor_current_state_service: SensorCurrentStateService , event_producer: EventDtoProducer):
    self._event_dto_producer: EventDtoProducer = event_producer
    self._sensor_current_state_service: SensorCurrentStateService = sensor_current_state_service

  def create(self, sensor_create_dto: SensorCreateDto) -> SensorCreatedDto:
    sensor, event = SensorFactory.create(sensor_create_dto.get_img(), sensor_create_dto.get_created_datetime())
    sensor_created_dto: SensorCreatedDto = SensorCreatedMapper().to_dto(sensor)
    event_dto: EventDto = EventMapper.to_dto(event)
    self._event_dto_producer.publish('sensor', event_dto)
    return sensor_created_dto

  def calculate(self, sensor_publish_data_dto: SensorPublishDataDto) -> None:
    sensor_id: str = sensor_publish_data_dto.get_sensor_id()
    img: str = sensor_publish_data_dto.get_img()
    place_id: str = sensor_publish_data_dto.get_place_id()
    sensor_current_state: SensorCurrentState
    print("sensor_id",sensor_id)
    sensor_created_dto: SensorCreatedDto = SensorCreatedDto(sensor_id)
    try:
      sensor_current_state = self._sensor_current_state_service.get_by_id(sensor_created_dto)
    except Exception as error:
      sensor_current_state = self._sensor_current_state_service.create(sensor_created_dto)
    
    sensor: Sensor = SensorFactory.instantiate(sensor_id)
    event: Event = sensor.calculate(sensor_current_state, place_id, img)
    
    event_dto: EventDto = EventMapper.to_dto(event)
    self._event_dto_producer.publish('sensor', event_dto)
    print('service received event! 🔥')