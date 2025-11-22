class Broker:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, topic, callback):
        """ Each topic will have a list of callbacks that get executed """
       
        # If the topic doesnt exist create an empty list 
        if topic not in self.subscribers:
            self.subscribers[topic] = []

        self.subscribers[topic].append(callback)

    def publish(self, topic, data=None):
        """ Takes topic name and optional data, calls each callback of the 
            topic with the data passed as arg"""

        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(data)