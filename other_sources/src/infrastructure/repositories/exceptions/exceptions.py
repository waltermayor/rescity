class NotFoundError(Exception):
    """To be raised when an element is not found on the repository."""
    pass

class SensorNotFoundError(NotFoundError):
    """To be raised when a sensor is not found on the repository."""
    def __init__(self) -> None:
        super().__init__(
            1001,
            "Resource not found",
            "Sensor doesn`t exist"
        )

