 
from producer import producer
from consumer import consumer
from kafka import KafkaConsumer
from dspDataPass import dspDataParse
import logging
import time
import json

class Main(object):
    def __init__(self):
        self.broker = ['ec2-52-23-230-199.compute-1.amazonaws.com:9092']
        self.producer = producer(self.broker)

    def run(self):
        starttime = time.time()
        consumer = KafkaConsumer('DSP', group_id='neural_nets', bootstrap_servers=self.broker)
        for message in consumer:
            data = json.dumps(str(message.value))
            self.producer.send_prediction(data)


if __name__ == "__main__":
    logging.info("Starting producer.....")
    main = Main()
    main.run()