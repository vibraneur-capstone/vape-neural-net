from kafka import KafkaConsumer
import time
# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer('DSP',
                         group_id='neural_nets',
                         bootstrap_servers=['ec2-52-23-230-199.compute-1.amazonaws.com:9092'])
                         
for message in consumer:
    #value = message.value]
    #Call the data conversion function here
    #@argument 
    #inputA
    #inputB
    print(value)
