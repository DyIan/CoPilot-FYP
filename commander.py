class Commander:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.keyboard_command = None
        
        self.features_commands = {}

    def update_player(self, new_player):
        self.player = new_player
    
    def register(self, broker):
        broker.subscribe("keyboard", self.update_keyboard)
        broker.subscribe("speed_limiter", lambda command: self.update_feature("speed_limiter", command))
        broker.subscribe("object", lambda command: self.update_feature("object", command))

    def update_keyboard(self, command):
        """ Game loop will call this to update the keyboard inputted control command"""
        self.keyboard_command = command

    def update_feature(self, name, command):
        self.features_commands[name] = command

    def apply_control(self):
        """ This is called when the loop has taken all commands and wants to move the car
        Later here we will do some calculation to decide which command to do"""
        command = self.keyboard_command

        if self.features_commands["speed_limiter"] != None:
            command.throttle = 0.0
        
        self.player.apply_control(command)

