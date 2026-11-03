
# Smart Parking & Vehicle Analytics System

An AI-powered computer vision project for parking occupancy monitoring, vehicle detection, multi-object tracking, traffic counting, and interactive analytics.

The system combines deep learning, computer vision, and data visualization to analyze parking-lot images and traffic videos.

## Features

### Parking Occupancy Analysis
- Parking-space classification using MobileNetV3-Small.
- Transfer learning using the CNRPark dataset.
- Parking-space localization using camera-specific coordinates.
- Occupied/free parking-space detection.
- Occupancy percentages and utilization statistics.
- Parking-slot utilization heatmaps.
- CSV analytics export.

### Vehicle Traffic Analysis
- Vehicle detection using YOLO11.
- Multi-object tracking using ByteTrack.
- Directional vehicle counting using a configurable counting line.
- Vehicle trajectory analysis.
- Movement duration and pixel-distance measurements.
- Annotated video generation.
- CSV analytics export.

### Interactive Dashboard
Built using Streamlit and Plotly.

The dashboard provides:
- Vehicle traffic statistics and visualizations.
- Parking occupancy statistics over time.
- Vehicle class distributions.
- Directional crossing counts.
- Parking-slot utilization analysis.
- Interactive tables and CSV downloads.
- Annotated traffic video playback.

## Technologies

- Python 3.12
- PyTorch and torchvision
- MobileNetV3-Small
- Ultralytics YOLO11
- ByteTrack
- OpenCV
- NumPy and pandas
- Matplotlib and Plotly
- Streamlit
- scikit-learn
- unittest

## Project Architecture

```text
smart-parking/
├── config/
│   └── vehicle_tracking.yaml
├── dashboard/
│   └── app.py
├── data/
│   └── public/                 # Datasets (not included)
├── results/
│   ├── analytics/              # Generated CSVs and charts
│   ├── models/                 # Trained model checkpoints
│   ├── parking/                # Annotated parking images
│   └── videos/                 # Annotated traffic videos
├── scripts/
│   ├── process_parking_folder.py
│   ├── process_vehicle_video.py
│   ├── train_classifier.py
│   └── ...
├── src/
│   ├── analytics/
│   ├── classification/
│   ├── common/
│   ├── detection/
│   ├── parking/
│   └── tracking/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Datasets

This project uses publicly available computer vision datasets.

### CNRPark / CNR-EXT

Used for parking-space classification and occupancy monitoring.

The dataset contains parking-space images captured under different environmental and camera conditions.

The parking occupancy pipeline processes full parking-lot images using predefined parking-space coordinates.

### AI City Challenge

Used for vehicle detection, tracking, and traffic analytics.

The development video contains:
- Resolution: 1280 × 960
- Frame rate: 10 FPS
- Duration: approximately 5 minutes
- Total frames: 3,000

Dataset files are not included in this repository.

Users must obtain the datasets separately and place them in the expected local directories.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fattah-alizadeh/smart-parking-vehicle-analytics.git
cd smart-parking-vehicle-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install a compatible CUDA-enabled PyTorch build using the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

The project was developed using Python 3.12.4 and a CUDA-enabled NVIDIA GPU.

## Running the Application

### Start the Streamlit dashboard

From the project root:

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal, normally:

http://localhost:8501

The dashboard currently visualizes previously generated analytics and video outputs.

It does not yet support uploading arbitrary videos through the interface.

### Process a traffic video

Configure the video processing settings in:

```text
config/vehicle_tracking.yaml
```

Then run:

```bash
python scripts/process_vehicle_video.py
```

The pipeline performs vehicle detection, tracking, directional counting, movement analysis, and analytics export.

### Process parking-lot images

Example:

```bash
python scripts/process_parking_folder.py "data/public/cnrpark/nrpark_full/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera1" "data/public/cnrpark/nrpark_full/CNR-EXT_FULL_IMAGE_1000x750/camera1.csv"
```

This requires the CNR-EXT dataset and the trained parking classifier checkpoint.

## Evaluation Results

### Vehicle Tracking and Counting

On the development traffic video:

| Metric | Result |
|---|---:|
| Processed frames | 3,000 |
| Raw tracking IDs | 1,016 |
| Filtered reliable tracks | 675 |
| Total detected line crossings | 181 |
| A to B crossings | 72 |
| B to A crossings | 109 |

Tracking IDs are not guaranteed to represent unique physical vehicles.

### Manual Counting Validation

A 30-second segment was manually reviewed.

| Direction | Manual | System |
|---|---:|---:|
| A to B | 3 | 3 |
| B to A | 15 | 12 |
| Total | 18 | 15 |

The system detected 15 of the 18 manually counted crossings, approximately 83.3%.

This is a limited validation sample, not a comprehensive benchmark.

### Parking Occupancy

For a sample of 21 images from one camera:

| Metric | Result |
|---|---:|
| Parking spaces | 35 |
| Peak occupied spaces | 32 |
| Minimum occupied spaces | 9 |
| Average occupancy | 68.03% |

## Automated Testing

The project includes 25 unit tests covering:

- Vehicle movement analysis
- Line crossing and directional counting
- Parking occupancy calculations
- CSV analytics export

Run all tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Latest local test result:

```text
Ran 25 tests

OK
```

## Limitations

- Parking-space coordinates are camera-specific.
- Parking classification accuracy may vary across cameras and weather conditions.
- Vehicle tracking may produce fragmented IDs.
- Line counting requires an appropriate line for the camera viewpoint.
- Dashboard outputs must be generated locally because datasets, model checkpoints, and generated videos are not included.
- Uploaded-video processing is not yet implemented.

## Future Improvements

- User-uploaded traffic video analysis.
- Interactive counting-line selection.
- Improved cross-camera parking classification.
- Additional vehicle tracking validation.
- Deployment and downloadable demonstration assets.

## Author

**Fattah Alizadeh**

GitHub: https://github.com/fattah-alizadeh

## License

No open-source license has been selected yet.
