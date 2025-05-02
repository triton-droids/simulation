import glfw  

class KeyboardController:
    def __init__(self, linear_speed=0.1, angular_speed=0.1):
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed

        # This dictionary keeps current velocity state
        self.command = {
            "linear_velocity": 0.0,
            "angular_velocity": 0.0
        }

        # Track which keys are currently pressed
        self.pressed_keys = set()

    def key_callback(self, window, key, scancode, action, mods):
        # Key press
        if action == glfw.PRESS:
            self.pressed_keys.add(key)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(key)

        self._update_command()

    def _update_command(self):
        # Reset
        self.command["linear_velocity"] = 0.0
        self.command["angular_velocity"] = 0.0

        if glfw.KEY_W in self.pressed_keys:
            self.command["linear_velocity"] += self.linear_speed
        if glfw.KEY_S in self.pressed_keys:
            self.command["linear_velocity"] -= self.linear_speed
        if glfw.KEY_A in self.pressed_keys:
            self.command["angular_velocity"] -= self.angular_speed
        if glfw.KEY_D in self.pressed_keys:
            self.command["angular_velocity"] += self.angular_speed

    def get_command(self):
        return self.command