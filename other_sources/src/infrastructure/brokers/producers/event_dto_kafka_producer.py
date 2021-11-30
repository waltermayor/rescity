from kafka import KafkaProducer
from bson import json_util
import os

from application.services.dtos.event_dto import EventDto, EventDtoProducer

class EventDtoKafkaProducer(EventDtoProducer):

  def __init__(self) -> None:
    self._kafka_producer: KafkaProducer = KafkaProducer(
      bootstrap_servers = [os.environ.get('KAFKA_HOST')+':'+os.environ.get('KAFKA_PORT')],
      value_serializer = lambda data: json_util.dumps(data.__dict__).encode("utf-8")
    )

  def publish(self, topic: str, event_dto: EventDto) -> None:
    print(f'>> Sending {event_dto.get_id()} to {topic} topic')
    self._kafka_producer.send(topic, event_dto)
