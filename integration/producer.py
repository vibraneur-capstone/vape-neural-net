from kafka import KafkaProducer
import json
class producer(object):
    def __init__(self, kafka_brokers):
        self.producer = KafkaProducer(
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            bootstrap_servers=kafka_brokers
        )
    def send_prediction(self, json_data):
        self.producer.send('neural-nets', json_data)
 