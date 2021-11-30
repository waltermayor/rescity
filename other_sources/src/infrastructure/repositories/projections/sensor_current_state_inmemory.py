
from typing import List
from application.services.projections.sensor_current_state import SensorCurrentStateRepository

from application.services.projections.sensor_current_state import SensorCurrentState
from src.infrastructure.repositories.exceptions.exceptions import SensorNotFoundError

class SensorCurrentStateInmemoryRepository(SensorCurrentStateRepository):

    def __init__(self):
        self._inmemory_database = {}

    
    def get_by_id(self, id: str) -> SensorCurrentState:
        sensor_current_state = self._inmemory_database.get(id)
        
     
        if sensor_current_state :
            return sensor_current_state

        else:
            raise SensorNotFoundError
   
        
    
    def get_all(self) -> List[SensorCurrentState]:
        raise NotImplementedError

    
    def save(self, sensor_current_state: SensorCurrentState) -> None:
        id = sensor_current_state.get_id()
        self._inmemory_database[id] = sensor_current_state

    def delete(self, sensor_current_state: SensorCurrentState) -> None:
        raise NotImplementedError