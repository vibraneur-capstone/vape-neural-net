from kafka import KafkaConsumer
from dspDataPass import dspDataParse
import time
class consumer(object):
    def __init__(self,kafka_brokers):
        consumer = KafkaConsumer('DSP',
                                group_id='neural_nets',
                                bootstrap_servers=kafka_brokers)
                         


