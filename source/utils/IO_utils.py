"""Small stdout and IO helpers used by scripts and notebooks."""

import sys
import os

class SuppressOutput:
    """ Silence stdout and stderr temporarily"""
    def __enter__(self):
        """Redirect stdout and stderr to the OS null device.

        Returns:
            None.

        Side effects:
            Replaces `sys.stdout` and `sys.stderr` until `__exit__` restores
            them.
        """

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, *args):
        """Restore stdout and stderr after a suppressed block.

        Args:
            *args: Exception details supplied by the context manager protocol.

        Side effects:
            Closes the temporary streams and restores the original streams.
        """

        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
