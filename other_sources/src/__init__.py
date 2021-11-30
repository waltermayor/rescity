import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

from flask import Flask
from flask_injector import FlaskInjector
from dependencies import configure
#from mongoengine import connect

# Driving Adapters
from interface.controllers.sensor_controller import sensor_controller
#from infrastructure.brokers.consumers.sensor_kafka_consumer import SensorKafkaConsumer



# TODO: Inject Mappers
# TODO: When to use mapper and when to instantiate a DTO?
# TODO: Projection as DTO?
# TODO: Flask Config Enviroments
# TODO: Error handdling
# TODO: Defensive programming
# TODO: Testing

def create_app():

  app = Flask (__name__)

  # Start MongoDB Connection
  """ connect(
    host=os.environ.get('MONGO_HOST'),
    port=int(os.environ.get('MONGO_PORT')),
    db=os.environ.get('MONGO_DB')
  ) """

  # Start Controllers
  app.register_blueprint(sensor_controller, url_prefix='/v1')

  flask_injector = FlaskInjector(app=app, modules=[configure])

  # Start Kafka Consumers
  #flask_injector.injector.get(SensorKafkaConsumer)

  return app