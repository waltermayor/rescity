
class PersonTrackedDto():

    def __init__(self, number_of_persons: int, number_of_close_persons: int, persons_without_mask: int) -> None:
        self.number_of_persons = number_of_persons
        self.number_of_close_persons = number_of_close_persons
        self.persons_without_mask = persons_without_mask

    def get_number_of_persons(self) -> int:
        return self.number_of_persons

    def get_number_of_close_persons(self) -> int:
        return self.number_of_close_persons

    def get_number_of_close_persons(self) -> int:
        return self.number_of_close_persons
