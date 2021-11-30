class ErrorDto():

  def __init__(self, code: int, message: str, description: str) -> None:
    self.code = code
    self.message = message
    self.description = description

  def get_code(self) -> str:
    return self.code

  def get_message(self) -> str:
    return self.message

  def get_description(self) -> str:
    return self.description