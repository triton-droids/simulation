import glfw  

class KeyboardController:
    def __init__(self, linear_speed=0.1, angular_speed=0.1):
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed

        # Track which keys are currently pressed
        self.pressed_keys = set()

        # Command format: [lin_vel_x, lin_vel_y, ang_vel_yaw]
        self.command = [0.0, 0.0, 0.0]

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.pressed_keys.add(key)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(key)

        self._update_command()

    def _update_command(self):
        lin_vel_x = 0.0
        lin_vel_y = 0.0  # Optional: support strafe with keys like Q/E or LEFT/RIGHT
        ang_vel_yaw = 0.0

        if glfw.KEY_W in self.pressed_keys:
            lin_vel_x += self.linear_speed
        if glfw.KEY_S in self.pressed_keys:
            lin_vel_x -= self.linear_speed
        if glfw.KEY_A in self.pressed_keys:
            ang_vel_yaw -= self.angular_speed
        if glfw.KEY_D in self.pressed_keys:
            ang_vel_yaw += self.angular_speed

        self.command = [lin_vel_x, lin_vel_y, ang_vel_yaw]

    def get_command(self):
        return self.command
