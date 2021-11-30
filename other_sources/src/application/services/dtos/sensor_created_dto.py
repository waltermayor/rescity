from bson.objectid import ObjectId, InvalidId

from application.services.dtos.exceptions.exceptions import InvalidIdError

class SensorCreatedDto():

  def __init__(self, id: str) -> None:
    try:
      ObjectId(id)
    except InvalidId:
      raise InvalidIdError

    self.id = id

  def get_id(self) -> str:
    return self.id