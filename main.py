"""
Basler Auto-Brightness Camera Capture Script with Digital Zoom
✅ Compatible with latest pypylon version (no AccessModeType)
"""

from pypylon import pylon as py
from pypylon import genicam
import cv2
import numpy as np
import time
from collections import deque


class BaslerAutoCamera:
    def __init__(self):
        self.camera = None
        self.converter = py.ImageFormatConverter()
        self.converter.OutputPixelFormat = py.PixelType_BGR8packed
        self.converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

        # Auto-brightness parameters
        self.target_brightness = 128
        self.brightness_tolerance = 10
        self.brightness_history = deque(maxlen=5)
        self.adjustment_step = 0.1
        self.min_exposure = 100
        self.max_exposure = 50000

        # Control flags
        self.running = False
        self.auto_adjust = True
        self.current_pixel_format = None
        self.zoom_factor = 1.0  # 1.0 = no zoom

    def initialize_camera(self):
        try:
            self.camera = py.InstantCamera(
                py.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()

            info = self.camera.GetDeviceInfo()
            print(f"Camera model : {info.GetModelName()}")
            print(f"Serial number: {info.GetSerialNumber()}")

            self.camera.Width.SetValue(self.camera.Width.Max)
            self.camera.Height.SetValue(self.camera.Height.Max)

            preferred = ["RGB8", "BGR8", "BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8", "Mono8"]
            for fmt in preferred:
                try:
                    if genicam.IsWritable(self.camera.PixelFormat):
                        available = [e.GetSymbolic() for e in self.camera.PixelFormat.GetEntries()]
                        if fmt in available:
                            self.camera.PixelFormat.SetValue(fmt)
                            print("Pixel format set to:", fmt)
                            break
                except Exception:
                    continue
            else:
                print("⚠ Preferred pixel formats unavailable.")
                print("Available:", [e.GetSymbolic() for e in self.camera.PixelFormat.GetEntries()])

            self.current_pixel_format = self.camera.PixelFormat.GetValue()

            # Set default exposure and gain
            self.camera.ExposureTime.SetValue(10000)
            self.camera.Gain.SetValue(self.camera.Gain.Min)

            # Disable auto features
            if genicam.IsWritable(self.camera.ExposureAuto):
                self.camera.ExposureAuto.SetValue("Off")
            if genicam.IsWritable(self.camera.GainAuto):
                self.camera.GainAuto.SetValue("Off")

            self.camera.AcquisitionMode.SetValue("Continuous")
            print("Camera initialized ✔")
            return True

        except Exception as e:
            print("❌ Error initializing camera:", e)
            return False

    def calculate_brightness(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.mean(gray)

    def adjust_exposure(self, current_brightness):
        if not self.auto_adjust:
            return
        self.brightness_history.append(current_brightness)
        avg_brightness = np.mean(self.brightness_history)
        diff = self.target_brightness - avg_brightness

        if abs(diff) > self.brightness_tolerance:
            try:
                current_exposure = self.camera.ExposureTime.GetValue()
                factor = 1.0 + (diff / 255.0) * self.adjustment_step
                new_exposure = max(self.min_exposure, min(self.max_exposure, current_exposure * factor))
                self.camera.ExposureTime.SetValue(new_exposure)
                print(f"Brightness: {avg_brightness:.1f}, Target: {self.target_brightness}, Exposure: {new_exposure:.0f} μs")
            except Exception as e:
                print("⚠ Error adjusting exposure:", e)

    def digital_zoom(self, img, zoom_factor):
        if zoom_factor <= 1.0:
            return img
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius_x, radius_y = int(w / (2 * zoom_factor)), int(h / (2 * zoom_factor))

        min_x, max_x = center_x - radius_x, center_x + radius_x
        min_y, max_y = center_y - radius_y, center_y + radius_y

        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(w, max_x), min(h, max_y)

        cropped = img[min_y:max_y, min_x:max_x]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def add_overlay(self, img, brightness, frame_count):
        try:
            exposure = self.camera.ExposureTime.GetValue()
            gain = self.camera.Gain.GetValue()
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            color = (0, 255, 0) if self.auto_adjust else (0, 0, 255)
            thickness = 2

            lines = [
                f"Frame: {frame_count}",
                f"Brightness: {brightness:.1f} (Target: {self.target_brightness})",
                f"Exposure: {exposure:.0f} μs",
                f"Gain: {gain:.1f}",
                f"Auto-adjust: {'ON' if self.auto_adjust else 'OFF'}",
                f"Zoom: {self.zoom_factor:.1f}x"
            ]

            for i, text in enumerate(lines):
                cv2.putText(img, text, (10, 30 + i * 25), font, scale, color, thickness)

        except Exception as e:
            print("⚠ Error adding overlay:", e)

    def save_single_image(self, img, frame_count):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"basler_capture_{timestamp}_frame{frame_count:06d}.jpg"
        cv2.imwrite(filename, img)
        print("📸 Image saved:", filename)

    def capture_continuous(self, display=True, save_images=False, save_interval=30):
        if not self.camera or not self.camera.IsOpen():
            print("⚠ Camera not initialized. Call initialize_camera() first.")
            return

        try:
            self.camera.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            self.running = True
            print("▶️ Starting continuous capture…")
            print("  [q] Quit | [a] Toggle auto-adjust | [s] Save image | [+/-] Adjust brightness | [ ] Zoom")

            frame_count = 0
            last_save_time = time.time()

            while self.running and self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(5000, py.TimeoutHandling_ThrowException)

                if grab_result.GrabSucceeded():
                    image = self.converter.Convert(grab_result)
                    img = image.GetArray()

                    brightness = self.calculate_brightness(img)
                    self.adjust_exposure(brightness)

                    img = self.digital_zoom(img, self.zoom_factor)

                    if display:
                        self.add_overlay(img, brightness, frame_count)
                        cv2.imshow("Basler Camera - Auto Brightness", img)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            break
                        elif key == ord('a'):
                            self.auto_adjust = not self.auto_adjust
                            print(f"Auto-adjust: {'ON' if self.auto_adjust else 'OFF'}")
                        elif key == ord('s'):
                            self.save_single_image(img, frame_count)
                        elif key in [ord('+'), ord('=')]:
                            self.target_brightness = min(240, self.target_brightness + 10)
                            print(f"Target brightness: {self.target_brightness}")
                        elif key == ord('-'):
                            self.target_brightness = max(50, self.target_brightness - 10)
                            print(f"Target brightness: {self.target_brightness}")
                        elif key == ord(']'):
                            self.zoom_factor = min(5.0, self.zoom_factor + 0.1)
                            print(f"🔍 Zoom factor: {self.zoom_factor:.1f}x")
                        elif key == ord('['):
                            self.zoom_factor = max(1.0, self.zoom_factor - 0.1)
                            print(f"🔍 Zoom factor: {self.zoom_factor:.1f}x")

                    if save_images and (time.time() - last_save_time > save_interval):
                        self.save_single_image(img, frame_count)
                        last_save_time = time.time()

                    frame_count += 1

                grab_result.Release()

        except Exception as e:
            print("❌ Error during capture:", e)
        finally:
            self.stop_capture()

    def stop_capture(self):
        self.running = False
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        cv2.destroyAllWindows()
        print("🛑 Capture stopped")

    def close_camera(self):
        if self.camera and self.camera.IsOpen():
            self.camera.Close()
        print("✅ Camera closed")

    def set_target_brightness(self, brightness):
        self.target_brightness = max(50, min(240, brightness))
        print(f"🎯 Target brightness set to: {self.target_brightness}")

    def set_brightness_tolerance(self, tolerance):
        self.brightness_tolerance = max(5, min(50, tolerance))
        print(f"📏 Brightness tolerance set to: {self.brightness_tolerance}")


def main():
    camera_app = BaslerAutoCamera()

    try:
        if not camera_app.initialize_camera():
            return

        camera_app.set_target_brightness(120)
        camera_app.set_brightness_tolerance(15)
        camera_app.capture_continuous(display=True, save_images=True, save_interval=60)

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print("❌ Application error:", e)
    finally:
        camera_app.close_camera()


if __name__ == "__main__":
    main()
