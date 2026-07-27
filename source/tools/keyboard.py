"""Keyboard input helpers used by interactive simulation tools."""

import glfw

class KeyboardController:
    """Convert WASD key state into a simple velocity command.

    Interactive tools can register `key_callback` with GLFW and read the latest
    `[lin_vel_x, lin_vel_y, ang_vel_yaw]` command through `get_command`.
    """

    def __init__(self, linear_speed=0.1, angular_speed=0.1):
        """Create a keyboard command mapper.

        Args:
            linear_speed: Forward/backward command increment.
            angular_speed: Yaw command increment.
        """

        self.linear_speed = linear_speed
        self.angular_speed = angular_speed

        # Track which keys are currently pressed
        self.pressed_keys = set()

        # Command format: [lin_vel_x, lin_vel_y, ang_vel_yaw]
        self.command = [0.0, 0.0, 0.0]

    def key_callback(self, window, key, scancode, action, mods):
        """Update pressed-key state from a GLFW callback.

        Args:
            window: GLFW window that received the input event.
            key: GLFW key code.
            scancode: Platform-specific scan code.
            action: GLFW press/release action.
            mods: Active modifier keys.

        Side effects:
            Updates the stored command.
        """

        if action == glfw.PRESS:
            self.pressed_keys.add(key)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(key)

        self._update_command()

    def _update_command(self):
        """Recompute the command vector from currently pressed keys.

        Side effects:
            Updates `self.command` using WASD key state.
        """

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
        """Return the latest keyboard velocity command.

        Returns:
            List in `[lin_vel_x, lin_vel_y, ang_vel_yaw]` format.
        """

        return self.command
