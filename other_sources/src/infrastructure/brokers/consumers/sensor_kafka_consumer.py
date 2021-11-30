from injector import inject
from kafka import KafkaConsumer
from bson import json_util
from threading import Thread
import os

from application.bounded_contexts.analysis.domain.model.sensor import SensorService
from application.services.dtos.event_dto import EventDto

class SensorKafkaConsumer:

  @inject
  def __init__(self, sensor_service: SensorService) -> None:

    # Set Handlers
    self._hanlders: dict = {
      'created': sensor_service.calculate
    }

    # Kafka Consumer Config
    self._kafka_consumer: KafkaConsumer = KafkaConsumer(
      'other_sources',
      bootstrap_servers = [os.environ.get('KAFKA_HOST')+':'+os.environ.get('KAFKA_PORT')],
      auto_offset_reset = 'earliest',
      group_id = 'other_sources_kafka_consumer',
      value_deserializer = lambda data: json_util.loads(data)
    )

    # Start Kafka Consumers as Threads
    Thread(
      target = self.__start_tread,
      daemon = True
    ).start()


  def __start_tread(self):
    for msg in self._kafka_consumer:
      topic: str = msg.topic
      event_dto: EventDto = EventDto(**msg.value)

      event_id: str = event_dto.get_id()
      event_type: str = event_dto.get_type()
      print(f'>> Consuming event {event_id} from {topic} with type {event_type}')

      try:
        self._hanlders[event_type](event_dto)
      except KeyError:
        print(f'>> Handler not implemented')
        pass
