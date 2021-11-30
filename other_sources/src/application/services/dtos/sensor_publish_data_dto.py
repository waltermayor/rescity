class SensorPublishDataDto():

  def __init__(self, place_id: str, sensor_id:str, img: str, created_datetime: int) -> None:
    self.sensor_id=sensor_id
    self.place_id=place_id
    self.img = img
    self.created_datetime = created_datetime

  def get_img(self) -> str:
    return self.img

  def get_sensor_id(self) -> str:
    return self.sensor_id

  def get_place_id(self) -> str:
    return self.place_id

  def get_created_datetime(self) -> str:
    return self.created_datetime
