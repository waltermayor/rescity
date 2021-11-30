class ConstrainError(Exception):
    """To be raised when an attribute doesn't match the constrains defined"""
    pass

class InvalidIdError(ConstrainError):
    """To be raised when attribute id doesn't match the longitude constrain defined"""
    def __init__(self) -> None:
        super().__init__(
            2001,
            "Constrain violated",
            "The \"id\" attribute is not 12 bytes nor 24 hex digits"
        )

class LongitudeNameError(ConstrainError):
    """To be raised when attribute name doesn't match the longitude constrain defined"""
    def __init__(self) -> None:
        super().__init__(
            2002,
            "Constrain violated",
            "The \"name\" attribute must be 50 character or less"
        )

class LongitudeEquationError(ConstrainError):
    """To be raised when attribute equation doesn't match the longitude constrain defined"""
    def __init__(self) -> None:
        super().__init__(
            2003,
            "Constrain violated",
            "The \"equation\" attribute must be 100 character or less"
        )

class InvalidEquationError(ConstrainError):
    """To be raised when attribute equation is invalid"""
    def __init__(self) -> None:
        super().__init__(
            2004,
            "Constrain violated",
            "The \"equation\" attribute is invalid"
        )

class LongitudeUnitsError(ConstrainError):
    """To be raised when attribute units doesn't match the longitude constrain defined"""
    def __init__(self) -> None:
        super().__init__(
            2005,
            "Constrain violated",
            "The \"units\" attribute must be 20 character or less"
        )
