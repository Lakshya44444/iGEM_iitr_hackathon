# iGEM_iitr_hackathon
Project Overview
This project represents my submission to the Tuberculosis Chest X-ray Classification Hackathon. The goal was to build a robust deep learning model capable of differentiating between Normal and TB-positive chest X-ray images.

Tuberculosis remains a major global health challenge. By achieving high diagnostic confidence, this system can serve as a vital fast-triage tool in clinical settings, particularly in resource-constrained areas where access to expert radiologists is limited.

Key Results and Performance
The model was rigorously optimized to maximize the competition's core metric.

Metric	Result	Description
Public Leaderboard Score (AUC)	[Insert Your Actual Score Here]	This score (Area Under the ROC Curve) measures the model's ability to rank positive cases higher than negative cases across all thresholds.
Model Architecture	ResNet50 (Transfer Learning)	Selected for its strong performance on image recognition tasks.
Prediction Technique	Test-Time Augmentation (TTA)	Used to stabilize predictions and boost final accuracy.

Export to Sheets
Technical Implementation Highlights
1. The Strategy: Focus on AUC
Because the submission was evaluated on AUC (requiring probability output), the model's final layer used a Sigmoid activation to output a float between 0.0 and 1.0. This probability reflects the model's confidence, which is essential for maximizing the metric.

2. High-Resolution Transfer Learning
To capture the subtle visual markers of early TB infection, the model was trained using higher-resolution inputs (384×384).

Backbone: ResNet50, initialized with ImageNet weights.

Normalization: We used ImageNet-specific preprocessing (preprocess_input) instead of simple division by 255. This is crucial for Transfer Learning success.

3. Combatting Data Issues
Medical datasets often suffer from imbalance, which can bias the model toward the dominant class (Normal).

Class Weighting: Applied custom weights during training to ensure the model penalized false negatives (missing a TB case) more heavily, forcing it to learn the minority class features effectively.

Data Augmentation: Used random rotations, shifts, and flips to create diverse training examples, preventing overfitting to the limited dataset size.

4. Code Robustness (The Fix)
A major challenge involved overriding TensorFlow's tendency to incorrectly infer a 1-channel (grayscale) input shape, which conflicted with the 3-channel (RGB) ImageNet weights. This was solved by explicitly setting color_mode='rgb' in the data generators, forcing the grayscale X-rays to be interpreted in the required 3-channel format.

Project Structure and Files
/Tuberculosis-CXR-Classifier
├── data/
│   ├── train/            # Training X-ray images
│   ├── test/             # Unlabeled images for prediction
│   ├── train_labels.csv  # Ground truth labels (0/1)
│   └── ...
├── src/
│   └── final_tb_classifier.py # <--- The full, final Python code
├── best_tb_resnet50_auc.h5  # Saved model weights
└── README.md              # This file
Installation and Execution
To run this project, you need a Python 3 environment with GPU support (recommended for speed).

Clone the Repository:

Bash

git clone https://github.com/YourUsername/RepoName.git
cd RepoName
Install Dependencies:

Bash

pip install tensorflow pandas numpy scikit-learn
Place Data: Ensure your train/, test/, and CSV files are correctly placed inside the ./data/ directory, mirroring the structure used in the code's BASE_DIR.

Run Training and Prediction:

Bash

python final_tb_classifier.py
(This script handles training, fine-tuning, TTA, and saving the final submission file.)

Future Work
Ensembling: Integrate predictions from other strong backbones (DenseNet201, EfficientNetB5) for a further stability and AUC boost.

Interpretability: Add a Grad-CAM analysis layer to visually highlight the regions of the X-ray that the model used to make its TB prediction, making it more trustworthy for clinicians.
