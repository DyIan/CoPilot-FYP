class Commander:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.keyboard_command = None
        self.software_command = None

    def update_player(self, new_player):
        self.player = new_player

    def update_keyboard(self, command):
        """ Game loop will call this to update the keyboard inputted control command"""
        self.keyboard_command = command

    def apply_control(self):
        """ This is called when the loop has taken all commands and wants to move the car
        Later here we will do some calculation to decide which command to do"""

        self.player.apply_control(self.keyboard_command)

