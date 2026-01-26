import carla
import math 

class Speed_Control:
    def __init__(self, broker):
        self.broker = broker
        self.speed_limit = 75
        broker.subscribe("speed_limit", self.speed_limit_callback)

    def speed_limit_callback(self, limit):
        """ Callback when topic 'speed_limit' published, sets the speed limit in Speed Control"""
        
        self.speed_limit = limit

    def speed_callback(self, velocity):
        """ When speed is published it converts it to km/h checks if its above speed limit 
            turns the throttle off if it is and publishes the command on 'speed_limiter' topic """
        
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if speed > self.speed_limit:
            #print(self.speed_limit)
            command = carla.VehicleControl(throttle=0.0)
            self.broker.publish("speed_limiter", command)
        else:
            self.broker.publish("speed_limiter", None)
        
