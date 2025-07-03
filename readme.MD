# Basler Auto-Brightness Camera Capture Script

## Overview

This Python script provides an automated camera capture solution for Basler cameras with intelligent brightness control. The application automatically adjusts camera exposure settings to maintain optimal image brightness while providing real-time video preview and image capture capabilities.

## Current Functionality

### Core Features
- **Auto-Brightness Control**: Automatically adjusts camera exposure based on real-time brightness analysis  
- **Real-time Preview**: Live video feed with overlay information  
- **Manual Image Capture**: Save individual images on demand  
- **Automatic Interval Capture**: Save images at specified intervals  
- **Interactive Controls**: Keyboard shortcuts for real-time parameter adjustment  
- **Digital Zoom Support**: Zoom in and out of the live preview using keyboard keys `[` and `]`  

### How Image Capture Works

#### Initialization Process
1. **Camera Detection**: Automatically detects and connects to the first available Basler camera  
2. **Resolution Setup**: Sets camera to maximum supported resolution  
3. **Pixel Format Configuration**: Attempts to set optimal pixel format (RGB8, BGR8, Bayer formats, or Mono8)  
4. **Exposure & Gain Setup**: Initializes with default exposure (10,000 μs) and minimum gain  
5. **Auto-Features Disabled**: Turns off camera's built-in auto-exposure and auto-gain for manual control  

#### Continuous Capture Loop
1. **Frame Acquisition**: Retrieves latest frame using `GrabStrategy_LatestImageOnly`  
2. **Image Conversion**: Converts camera-specific format to BGR8 for OpenCV compatibility  
3. **Brightness Analysis**: Calculates average brightness of the entire frame  
4. **Exposure Adjustment**: Modifies exposure time based on brightness deviation from target  
5. **Overlay Addition**: Adds real-time information display on the preview  
6. **Digital Zoom (Optional)**: Crops and scales image to simulate zoom based on `zoom_factor`  
7. **Display/Save**: Shows preview window and optionally saves images  

#### Brightness Control Algorithm
- **Target Brightness**: Default 128 (adjustable 50–240)  
- **Tolerance**: ±15 brightness units before adjustment  
- **History Tracking**: Uses rolling average of last 5 frames for stability  
- **Adjustment Formula**:  
  `new_exposure = current_exposure × (1 + brightness_diff / 255 × 0.1)`  
- **Exposure Limits**: Constrained between 100–50,000 μs  

### Current Capture Settings
- **Default Target Brightness**: 120  
- **Brightness Tolerance**: 15 units  
- **Auto-save Interval**: 60 seconds  
- **Image Format**: JPEG with timestamp and frame number  
- **Zoom Range**: 1.0x to 5.0x (adjustable during live preview)  

## Controls

| Key     | Action                                       |
|---------|----------------------------------------------|
| `q`     | Quit application                             |
| `a`     | Toggle auto-brightness adjustment            |
| `s`     | Save current frame manually                  |
| `+/=`   | Increase target brightness (+10)             |
| `-`     | Decrease target brightness (-10)             |
| `[`     | Zoom out (decrease digital zoom factor)      |
| `]`     | Zoom in (increase digital zoom factor)       |

## Technical Specifications

### Dependencies
- `pypylon` - Basler camera SDK  
- `opencv-python` - Image processing and display  
- `numpy` - Numerical operations  

### Camera Configuration
- **Acquisition Mode**: Continuous  
- **Pixel Format Priority**: RGB8 → BGR8 → Bayer → Mono8  
- **Resolution**: Maximum supported by camera  
- **Exposure Range**: 100–50,000 μs  
- **Gain**: Set to minimum initially  

## Issues and Recommendations

### 🔍 Zoom Discrepancy Issue

**Problem Observed**: The live preview window shows excessive zoom compared to the actual captured images.

**Evidence**:  
- Captured image (Image 1) shows proper field of view with correct framing  
- Live preview (Image 2) appears significantly zoomed in, showing only a portion of the scene  

**Potential Causes**:
1. **OpenCV Window Scaling**: The `cv2.imshow()` function may be auto-scaling the display window  
2. **Image Format Conversion**: Pixel format conversion might be cropping the image  
3. **Display Resolution Mismatch**: Monitor resolution affecting preview scaling  

**Recommended Solutions**:
```python
# Add these modifications to the display section:

# 1. Control window size explicitly
cv2.namedWindow("Basler Camera - Auto Brightness", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Basler Camera - Auto Brightness", 800, 600)

# 2. Add image scaling option
display_scale = 0.5  # Adjust as needed
display_img = cv2.resize(img, None, fx=display_scale, fy=display_scale)
cv2.imshow("Basler Camera - Auto Brightness", display_img)

# 3. Add debug information
print(f"Original image size: {img.shape}")
print(f"Camera resolution: {self.camera.Width.GetValue()}x{self.camera.Height.GetValue()}")
