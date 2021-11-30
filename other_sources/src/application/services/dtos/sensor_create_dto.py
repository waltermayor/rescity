class SensorCreateDto():

  def __init__(self, img: str, created_datetime: int) -> None:
    self.img = img
    self.created_datetime = created_datetime

  def get_img(self) -> str:
    return self.img

  def get_created_datetime(self) -> int:
    return self.created_datetime
