import carla
import math 

class Speed_Control:
    def __init__(self, broker):
        self.broker = broker
        self.speed_limit = 75
        broker.subscribe("speed_limit", self.speed_limit_callback)

    def speed_limit_callback(self, limit):
        self.speed_limit = limit

    def speed_callback(self, velocity):
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if speed > self.speed_limit:
            print(self.speed_limit)
            command = carla.VehicleControl(throttle=0.0)
            self.broker.publish("speed_limiter", command)
        else:
            self.broker.publish("speed_limiter", None)
        
