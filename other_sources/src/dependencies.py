from injector import singleton

# Ports
from application.bounded_contexts.analysis.domain.model.sensor import SensorService
from application.services.dtos.event_dto import EventDtoProducer

# Adapters
from application.services.sensor_service_v1 import SensorServiceV1
from infrastructure.brokers.producers.event_dto_kafka_producer import EventDtoKafkaProducer
from src.application.services.projections.sensor_current_state import SensorCurrentStateRepository, SensorCurrentStateService
from src.application.services.sensor_current_state_service_v1 import SensorCurrentStateServiceV1
from src.infrastructure.repositories.projections.sensor_current_state_inmemory import SensorCurrentStateInmemoryRepository

def configure(binder):
    binder.bind(SensorService, to=SensorServiceV1, scope=singleton)
    binder.bind(EventDtoProducer, to=EventDtoKafkaProducer, scope=singleton)
    binder.bind(SensorCurrentStateService, to=SensorCurrentStateServiceV1, scope=singleton)
    binder.bind(SensorCurrentStateRepository, to=SensorCurrentStateInmemoryRepository, scope=singleton)
