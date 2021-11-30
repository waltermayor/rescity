import time

from application.bounded_contexts.analysis.domain.model._event import Event
from application.services.dtos.event_dto import EventDto

class EventMapper:

  @staticmethod
  def to_dto(event: Event) -> EventDto:
    return EventDto(
      str(event.get_id()),
      event.get_type(),
      str(event.get_aggregate_id()),
      event.get_aggregate_type(),
      event.get_data(),
      int(time.mktime(event.get_created_datetime().timetuple()))
    )
