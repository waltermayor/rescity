from application.bounded_contexts.analysis.domain.model.sensor import Sensor
from application.services.dtos.sensor_created_dto import SensorCreatedDto

class SensorCreatedMapper:

  @staticmethod
  def to_dto(sensor: Sensor) -> SensorCreatedDto:
    return SensorCreatedDto(
      str(sensor.get_id())
    )