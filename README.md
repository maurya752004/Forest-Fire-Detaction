# Forest Fire Detection

A YOLO-based deep learning model for detecting forest fires in video streams with real-time inference and annotation.

## Project Overview

This project uses YOLOv8 classification model to detect fire and non-fire frames in video data. The model processes video frames, classifies each frame as Fire or Non_Fire, and generates annotated output videos with confidence scores.

## Features

- **Real-time fire detection** in video streams
- **YOLO-based classification** (YOLOv8 architecture)
- **Frame-by-frame analysis** with confidence scoring
- **Annotated video output** with predictions overlaid
- **Per-frame prediction logs** for detailed analysis
- **Fire frame extraction** - saves detected fire frames as JPEG images
- **Confidence boosting** for improved fire detection sensitivity
- **Preprocessed input** - RGB conversion and frame resizing (224x224)

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- ffmpeg (for video processing, optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd forest_fire_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
forest_fire_detection/
├── test_video.py                          # Main inference script
├── mainn.py                               # Secondary script (legacy)
├── fire_detection_model.h5                # Keras model file
├── fire_detection_model.keras             # Alternative model format
├── fire_detection_yolo/
│   └── yolo_fire_v1_cls2/
│       └── weights/
│           └── best.pt                    # YOLOv8 best weights
├── archive/
│   └── Data/
│       ├── Test_Data/
│       │   ├── Fire/
│       │   └── Non_Fire/
│       └── Train_Data/
│           ├── Fire/
│           └── Non_Fire/
├── dataset/
│   └── images/
│       ├── train/
│       │   ├── Fire/
│       │   └── Non_Fire/
│       └── val/
│           ├── Fire/
│           └── Non_Fire/
└── README.md
```

## Usage

### Basic Video Processing

```python
from test_video import process_video
from pathlib import Path

# Process a video
model_path = "fire_detection_yolo/yolo_fire_v1_cls2/weights/best.pt"
video_path = "path/to/video.mp4"
output_dir = Path("output2")

process_video(model_path, video_path, output_dir)
```

### Command Line Usage

```bash
python test_video.py
```

This will process the default video and save outputs to `videooutput2/<timestamp>/`.

## Output Files

After processing a video, the following files are generated:

1. **Annotated Video**: `processed_<original_filename>.mp4`
   - Original video with bounding boxes and predictions overlaid
   - Fire frames marked with red overlay
   - Non-fire frames marked with green overlay

2. **Prediction Log**: `predictions.txt`
   - Frame-by-frame analysis with confidence scores
   - Format: `Frame <N>: <Class> - Confidence: <score>`

3. **Fire Frames**: `fire_frame_<N>.jpg`
   - Individual JPEG images of detected fire frames
   - Useful for manual review and validation

## Model Details

- **Architecture**: YOLOv8 Classification
- **Input Size**: 224x224 pixels
- **Classes**: 2 (Fire, Non_Fire)
- **Model Path**: `fire_detection_yolo/yolo_fire_v1_cls2/weights/best.pt`
- **Framework**: Ultralytics YOLO

### Preprocessing

- Color space conversion: BGR → RGB
- Image resizing: Original → 224x224 (linear interpolation)
- Normalization: Applied by YOLO model

### Inference Settings

- Image size: 224x224
- Confidence threshold: 0.2 (for fire detection)
- Fire confidence boost: 1.2x (capped at 1.0)
- Non-fire confidence scaling: 0.9x

## Performance Notes

- **Processing Speed**: ~2-3ms per frame (depends on hardware)
- **GPU Support**: Recommended for faster processing
- **Video Formats**: MP4, AVI, MOV supported (via OpenCV)

## Dependencies

- `ultralytics>=8.0.0` - YOLO model framework
- `opencv-python>=4.5.0` - Video processing and annotation
- `numpy` - Numerical operations

See `requirements.txt` for full list.

## Training Data

The model was trained on:
- **Fire frames**: Images of forest fires, smoke, flames
- **Non-fire frames**: Forest scenes, non-fire landscapes
- **Train/Test split**: Available in `dataset/` directory

## Example Results

### Fire Detection (stream.mp4)
- Successfully identifies frames with visible fire/smoke
- High confidence for clear fire indicators

### Non-fire Detection (4K Forest Video)
- Correctly classifies dense forest scenes as non-fire
- Handles various lighting conditions and vegetation

## Improving Detection

### Confidence Adjustment
Modify the confidence boosting factors in `test_video.py`:
```python
if class_name == "Fire":
    confidence = min(confidence * 1.2, 1.0)  # Boost fire detection
else:
    confidence = confidence * 0.9
```

### Temporal Smoothing (Future Enhancement)
Consider implementing rolling-average filtering over N frames to reduce frame-to-frame noise.

### Custom Threshold
Adjust the detection threshold in `process_video()` function:
```python
if class_name == "Fire" and confidence > 0.2:  # Change 0.2 to desired threshold
    # Mark as fire
```

## Troubleshooting

### Video Processing Errors
- Ensure video file exists and is readable
- Check that output directory exists or can be created
- Verify ffmpeg is installed for video encoding

### Model Loading Issues
- Confirm model path is correct
- Ensure YOLO weights file (`.pt`) is in the correct location
- Check PyTorch/Ultralytics compatibility

### Memory Issues with Large Videos
- Process videos in chunks or segments
- Reduce resolution if needed
- Monitor GPU/RAM usage during processing

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is provided as-is for forest fire detection research and monitoring purposes.

## Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Acknowledgments

- YOLOv8 framework by Ultralytics
- OpenCV for video processing
- Forest fire detection research community

---

**Last Updated**: November 2025
**Model Version**: YOLO Fire v1 (2 classes)
**Status**: Active Development
