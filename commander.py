import math

class Commander:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.keyboard_command = None
        self.last_steer = 0.0
        
        self.features_commands = {}

    def update_player(self, new_player):
        """ A callback for when topic player_change publishes. Usually when vehicle is changed """

        self.player = new_player
    

    def register(self, broker):
        """ The subscriptions to the topic and their callbacks happen here"""

        broker.subscribe("keyboard", self.update_keyboard)
        broker.subscribe("speed_limiter", lambda command: self.update_feature("speed_limiter", command))    # Lambda takes the command as and arg so will call update_feature(speed_limiter, command)
        broker.subscribe("object", lambda command: self.update_feature("object", command))
        broker.subscribe("steering_error", lambda error: self.update_feature("steering_error", error))
        broker.subscribe("speed", lambda speed: self.update_feature("speed", speed))
        broker.subscribe("solid_intrusion", lambda error: self.update_feature("solid_intrusion", error))


    def update_keyboard(self, command):
        """ Game loop will publish to keyboard topic to update the keyboard inputted control command """

        self.keyboard_command = command


    def update_feature(self, name, command):
        """ Added to a dictionary that stores all the features desired controls """

        self.features_commands[name] = command

    def update_steering(self, drivers_steer):
        #steering_error = self.features_commands["steering_error"]
        steer = float(drivers_steer)

        intrusion = self.features_commands.get("solid_intrusion", 0.0)
        if abs(intrusion) < 1e-6:
            self.last_steer = steer
            return None, 1.0

        DEADZONE = 15.0
        K = 0.002
        max_steer = 0.12
        SMOOTH_ALPHA = 0.15
        HARD_PX = 100.0 
        EMERGENCY_PX = 200.0 
        throttle_power = 1.0
        MAX_OVERRIDE = 0.10 

        # Soft assist
        if abs(intrusion) <= DEADZONE:
            target = 0.0
        else:
            # +intrusion = push right, -intrusion = push left
            target = max_steer * math.tanh(K * (intrusion - math.copysign(DEADZONE, intrusion)))

        steer = steer + target
        steer = (1 - SMOOTH_ALPHA) * self.last_steer + SMOOTH_ALPHA * steer
        self.last_steer = steer

        # HARD BLOCK. If too close to road boundary disable steering into it
        if intrusion > HARD_PX and steer < 0.0:     # Need to go right
            steer = max(steer, 0.0) # block left steer
            throttle_power = 0.5    
            #print("HARD BLOCK OF LEFT")
        elif intrusion < -HARD_PX and steer > 0.0:  # Need to go left
            steer = min(steer, 0.0) # block right steer
            throttle_power = 0.5
            #print("HARD BLOCK OF RIGHT")


        # EMERGENCY BLOCK: strong push otherway
        if intrusion > EMERGENCY_PX:
            steer = max(steer, +MAX_OVERRIDE)
            throttle_power = 0.0
            #print("Emergency BLOCK OF LEFT")
        elif intrusion < -EMERGENCY_PX:
            steer = min(steer, -MAX_OVERRIDE)
            throttle_power = 0.0
            #print("Emergency BLOCK OF RIGHT")
        
        
        # Smooth it
        steer = float(max(-1.0, min(1.0, steer)))

        return steer, throttle_power
    
    def get_speed(self):
        speed = self.features_commands["speed"]
        converted_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)
        return converted_speed


    def apply_control(self):
        """ This is called when the loop has taken all commands and wants to move the car
        Later here we will do some calculation to decide which command to do"""
        command = self.keyboard_command 
        if command is None:     # Effects update_steering if None
            return
        
        speed = self.get_speed()
        steering_command, throttle = self.update_steering(command.steer)
        
        
        if steering_command is not None:    # and speed > 20
            command.steer = steering_command
            #command.throttle = throttle * command.throttle

        if self.features_commands["speed_limiter"] != None:
            command.throttle = 0.0

        
        self.player.apply_control(command)

