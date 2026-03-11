import mss
import numpy as np
from AppKit import NSEvent, NSScreen
from Quartz import CGEventSourceCreate, kCGEventSourceStateCombinedSessionState

class ScreenSampler:
    def __init__(self):
        self.sct = mss.mss()

    def get_scale_factor(self):
        """
        Gets the backing scale factor (Retina = 2.0, Standard = 1.0).
        """
        return NSScreen.mainScreen().backingScaleFactor()

    def get_mouse_position(self):
        """
        Natively gets the mouse position on macOS using Quartz.
        Returns (x, y) logical coordinates.
        """
        loc = NSEvent.mouseLocation()
        # macOS has origin at bottom-left, MSS expects top-left.
        # Use the screen height of the screen containing the mouse
        screen = NSScreen.mainScreen()
        height = screen.frame().size.height
        return int(loc.x), int(height - loc.y)

    def sample_pixel(self, x=None, y=None):
        """
        Samples the color of a pixel at (x, y).
        If (x, y) is None, samples at current mouse position.
        Handles Retina display scaling (e.g., 2.0x on MacBook Pro).
        """
        if x is None or y is None:
            x, y = self.get_mouse_position()
        
        scale = self.get_scale_factor()
        
        # Convert logical coordinates to physical pixels
        phys_x = int(x * scale)
        phys_y = int(y * scale)
        
        # MSS uses physical coordinates. 
        # On a 1440x900 logical screen with 2x scaling, (10, 10) is (20, 20) in MSS.
        region = {"top": phys_y, "left": phys_x, "width": 1, "height": 1}
        img = self.sct.grab(region)
        
        # Convert to RGB (MSS is BGRA)
        color = np.array(img.pixel(0, 0))[:3][::-1] 
        
        return tuple(color)

    def to_hex(self, rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

if __name__ == "__main__":
    # Test
    sampler = ScreenSampler()
    try:
        print("Move your mouse and press Ctrl+C to stop...")
        import time
        while True:
            pos = sampler.get_mouse_position()
            color = sampler.sample_pixel()
            hex_color = sampler.to_hex(color)
            print(f"Pos: {pos} | Color: {color} | Hex: {hex_color}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
