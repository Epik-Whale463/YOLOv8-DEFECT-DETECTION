
# Metallic Surface Defect Detection using YOLOv8

![Defect Detection](https://img.shields.io/badge/Computer%20Vision-Defect%20Detection-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-brightgreen)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This repository contains a deep learning model for detecting and classifying defects on metallic surfaces using YOLOv8. The model is trained to identify six types of common defects:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

This solution can be integrated into manufacturing quality control systems to automate defect detection, improving efficiency and reducing errors in quality inspection processes.

## 🔍 Dataset

The project uses the Metallic Surface Defect dataset, which contains images of various defects commonly found in metallic surfaces during manufacturing. The dataset includes:

- 555 annotated defect instances across 295 validation images
- 6 defect classes with varying frequencies
- Annotations in YOLO format

## 🧠 Model Architecture

- Base model: YOLOv8n (nano version)
- Input size: 640×640
- Training strategy: Transfer learning from pre-trained weights

## 🔧 Training Details

The model was trained on Kaggle using a Tesla P100 GPU with the following configuration:

- **Epochs**: 50
- **Batch Size**: 16
- **Optimizer**: Default YOLOv8 optimizer
- **Augmentations**: 
  - Mosaic (auto-disabled in last epochs)
  - Blur and MedianBlur (p=0.01)
  - ToGray (p=0.01)
  - CLAHE (p=0.01)

## 📊 Performance

### Model Metrics (mAP50-95)

| Class           | Precision | Recall | mAP50 | mAP50-95 |
|-----------------|-----------|--------|-------|----------|
| All             | 0.583     | 0.641  | 0.623 | 0.323    |
| Crazing         | 0.368     | 0.345  | 0.298 | 0.113    |
| Inclusion       | 0.615     | 0.743  | 0.737 | 0.375    |
| Patches         | 0.669     | 0.837  | 0.814 | 0.499    |
| Pitted Surface  | 0.728     | 0.622  | 0.635 | 0.364    |
| Rolled-in Scale | 0.512     | 0.486  | 0.520 | 0.255    |
| Scratches       | 0.606     | 0.811  | 0.732 | 0.335    |

![WhatsApp Image 2025-05-07 at 17 42 07_208903c7](https://github.com/user-attachments/assets/ebd1b303-f832-423c-82fc-4b4e5af26d5f)

### F! curve

![WhatsApp Image 2025-05-07 at 17 42 26_f3f4eab7](https://github.com/user-attachments/assets/b16349cc-b32f-4222-9ba4-f91424907976)


### Inference Speed
- Preprocess: 0.2ms
- Inference: 1.9ms
- Postprocess: 2.4ms per image

## 📈 Training Progress

The model showed consistent improvement throughout training:
- Early epochs showed rapid progress in defect classification
- Loss values steadily decreased over time
- mAP50 stabilized around 0.62 by the end of training

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone this repository
git clone https://github.com/Epik-Whale/YOLOv8-DEFECT-DETECTION.git
cd metallic-defect-detection

# Install dependencies
pip install -r requirements.txt
```

### Using the Model

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('path/to/best.pt')

# Perform inference on an image
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

## 🔮 Future Improvements

- Implement ensemble methods to improve defect detection accuracy
- Explore transfer learning with larger YOLOv8 variants
- Add a post-processing pipeline for deployment in manufacturing environments
- Create a web UI for easy interaction with the model
- Fine-tune hyperparameters to improve detection of challenging defect classes (e.g., crazing)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [Kaggle](https://www.kaggle.com) for providing the computational resources
- [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkc/en/zsresource.html) which inspired this project

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact me at:
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
