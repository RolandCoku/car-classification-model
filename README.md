# Automated Car Image Classification and Segmentation


## Project Overview

This project applies deep-learning methods to image management in an online car
marketplace. It contains two computer-vision tasks:

1. **Image classification:** classify an uploaded car photograph as either
   `interior` or `exterior`.
2. **Semantic segmentation:** identify the pixels belonging to a car and separate
   the vehicle from its background.

The complete implementation, training process, evaluation, visualizations, and
model export steps are contained in
`car_view_classification_and_segmentation.ipynb`.

## Motivation

The idea for this project comes from a personal car-marketplace application that I
am developing. Sellers upload multiple photographs for each vehicle listing, but
these photographs are not automatically organized by view type.

An interior/exterior classifier can automatically categorize each uploaded image.
This makes it possible to organize listing galleries, select an exterior photograph
as the cover image, and present interior photographs in a separate section.

The segmentation component supports the same application by extracting the vehicle
from its background. The resulting mask can be used to create cleaner and more
consistent listing images.

## Classification Task

### Dataset

The classifier uses `Dinusharg/Car_interior_exterior_v1`, a dataset created for
binary classification of car interior and exterior photographs. The preparation
pipeline:

- downloads the dataset from Hugging Face;
- verifies that image files can be opened;
- removes exact duplicate images;
- organizes files into `interior` and `exterior` classes;
- creates deterministic and stratified train, validation, and test sets.

The final split uses:

- **70%** for training;
- **15%** for validation;
- **15%** for testing.

### Model

The classification model uses MobileNetV2 pretrained on ImageNet. Training is
performed in two stages:

1. The MobileNetV2 feature extractor is frozen while a new binary classification
   head is trained.
2. The final MobileNetV2 layers are fine-tuned with a lower learning rate.

Data augmentation applies horizontal flips, small rotations, zoom, and contrast
changes during training.

### Evaluation

The classifier is evaluated using:

- test accuracy;
- precision, recall, and F1-score for each class;
- confusion matrix;
- training and validation accuracy curves;
- training and validation loss curves;
- predictions on an independent set of 14 car photographs.

The notebook also includes a `classify_image()` function that demonstrates how the
trained model can be integrated into the image-upload workflow of the marketplace.

## Segmentation Task

### Dataset

The segmentation model uses the Carvana Image Masking dataset. Each training
photograph is paired with a ground-truth binary mask that identifies the car.

Images and masks are resized to `128 × 128` pixels. Masks are converted to binary
values before training.

### Model

The segmentation model is a U-Net implemented with TensorFlow and Keras. It contains:

- an encoder that extracts visual features;
- a bottleneck that represents the image at its most compact level;
- a decoder that reconstructs the segmentation mask;
- skip connections that preserve spatial details.

The model is trained using a combined binary cross-entropy and Dice loss.

### Evaluation

The segmentation model is evaluated using:

- Dice coefficient;
- Intersection over Union;
- training and validation loss curves;
- visual comparison of the input image, ground-truth mask, predicted mask, and
  prediction overlay.

## Repository Structure

```text
car-view-classifier/
├── car_view_classification_and_segmentation.ipynb
├── build_notebook.py
├── requirements.txt
├── README.md
└── assets/
    └── ood_test/
```

The downloaded datasets and trained models are stored in `data/` and `models/`.
These directories are generated when the notebook is executed.

## Running the Project

### Google Colab

1. Open `car_view_classification_and_segmentation.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Accept access to `Dinusharg/Car_interior_exterior_v1` on Hugging Face.
4. Run the notebook cells in order and provide a Hugging Face read token when
   requested.
5. Upload the 14 files from `assets/ood_test/` during the independent classifier
   evaluation.
6. Provide Kaggle credentials when the segmentation dataset is downloaded.

### Local Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxxxx
jupyter notebook car_view_classification_and_segmentation.ipynb
```

Kaggle credentials are required when executing the segmentation section.

## Technologies

- Python
- TensorFlow and Keras
- MobileNetV2 transfer learning
- U-Net semantic segmentation
- NumPy
- scikit-learn
- Matplotlib
- Pillow
