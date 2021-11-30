from flask import Blueprint, request, Response
from bson import json_util
from injector import inject

from application.services.dtos.sensor_create_dto import SensorCreateDto
from application.services.dtos.sensor_publish_data_dto import SensorPublishDataDto 
from application.services.dtos.error_dto import ErrorDto
from application.bounded_contexts.analysis.domain.model.sensor import SensorService

sensor_controller = Blueprint('sensor_controller', __name__)

@inject
@sensor_controller.route('/publish-data', methods=['POST'])
def publish_data(sensor_service: SensorService):
  img: str = request.json['img']
  created_datetime: str = request.json['created_datetime']
  place_id: int = request.json['place_id']
  sensor_id: int = request.json['sensor_id']

  try:
    sensor_publish_data_dto: SensorPublishDataDto = SensorPublishDataDto(place_id, sensor_id, img, created_datetime)
  except Exception as error:
    code, message, description = error.args
    error_dto: ErrorDto = ErrorDto(code, message, description)
    error = json_util.dumps(error_dto.__dict__)
    return Response(error, mimetype = 'application/json', status = 400)

  sensor_service.calculate(sensor_publish_data_dto)
  return Response(mimetype = 'application/json', status = 202)
