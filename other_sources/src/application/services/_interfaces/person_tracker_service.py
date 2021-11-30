from abc import ABCMeta, abstractmethod
from application.services.dtos.event_dto import EventDto 
from application.services.dtos.person_tracked_dto import PersonTrackedDto

class PersonTrackerService(metaclass=ABCMeta):

  @abstractmethod
  def analyseImage(self, event_dto: EventDto) -> PersonTrackedDto:
    raise NotImplementedError
