
import time
import cv2
import numpy as np
from onvif import ONVIFCamera
from pynput import keyboard
from config import settings

class PTZController:
    def __init__(self, ip, port, username, password):
        self.camera = ONVIFCamera(ip, port, username, password)
        self.media = self.camera.create_media_service()
        self.ptz = self.camera.create_ptz_service()
        self.profile = self.media.GetProfiles()[0]

        if not self.profile.PTZConfiguration:
            raise RuntimeError("PTZ is not configured in this camera profile.")

        self.req = self.ptz.create_type('ContinuousMove')
        self.req.ProfileToken = self.profile.token

    def move(self, x, y):
        """Move in the given x (pan) and y (tilt) direction continuously."""
        self.req.Velocity = {
            'PanTilt': {
                'x': x,
                'y': y,
                'space': 'http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace'
            }
        }
        self.ptz.ContinuousMove(self.req)
        print(f"[PTZ] Moving camera - x: {x}, y: {y}")

    def stop(self):
        """Stop any camera movement."""
        stop = self.ptz.create_type('Stop')
        stop.ProfileToken = self.profile.token
        stop.PanTilt = True
        stop.Zoom = False
        self.ptz.Stop(stop)
        print("[PTZ] Camera movement stopped.")


    def control_loop(self):
        print("Arrow keys to move camera. Press 's' to stop, 'q' to quit.")

        def on_press(key):
            try:
                if key.char == 'q':
                    self.stop()
                    return False  # Stop listener
                elif key.char == 's':
                    self.stop()
            except AttributeError:
                if key == keyboard.Key.left:
                    self.move(-1.0, 0)
                    time.sleep(0.5)
                    self.stop()
                elif key == keyboard.Key.right:
                    self.move(1.0, 0)
                    time.sleep(0.5)
                    self.stop()
                elif key == keyboard.Key.up:
                    self.move(0, 1.0)
                    time.sleep(0.5)
                    self.stop()
                elif key == keyboard.Key.down:
                    self.move(0, -1.0)
                    time.sleep(0.5)
                    self.stop()

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

def run_ptz_if_enabled(enable_control=True):
    if enable_control:
        ptz = PTZController(ip=settings.CAMERA_IP, port=80, username=settings.CAMERA_USERNAME, password=settings.CAMERA_PASSWORD)
        ptz.control_loop()
    else:
        print("PTZ control disabled.")

if __name__ == '__main__':
    run_ptz_if_enabled(enable_control=True)
