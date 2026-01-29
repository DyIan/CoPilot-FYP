class Commander:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.keyboard_command = None
        
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


    def update_keyboard(self, command):
        """ Game loop will publish to keyboard topic to update the keyboard inputted control command """

        self.keyboard_command = command


    def update_feature(self, name, command):
        """ Added to a dictionary that stores all the features desired controls """

        self.features_commands[name] = command

    def update_steering(self):
        steering_amount = .5
        steering_error = self.features_commands["steering_error"]
        if steering_error > 200:
            target = +steering_amount
        elif steering_error < -200:
            target = -steering_amount
        else:
            target = None

        return target



    def apply_control(self):
        """ This is called when the loop has taken all commands and wants to move the car
        Later here we will do some calculation to decide which command to do"""

        steering_command = self.update_steering()
        command = self.keyboard_command

        if self.features_commands["speed_limiter"] != None:
            command.throttle = 0.0

        if steering_command is not None:
            command.steer = steering_command
    
        self.player.apply_control(command)

