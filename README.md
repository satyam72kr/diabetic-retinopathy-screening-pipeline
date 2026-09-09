# Automated Diabetic Retinopathy Diagnostic Pipeline via Transfer Learning

An advanced ophthalmology screening system designed to automate the classification and grading of retinal fundus photographs.

## Key Highlights & Metrics
* **Transfer Learning Backbone:** Leveraged a deep **ResNet50 architecture** to extract intricate vascular features from complex eye scans.
* **Clinical Specificity:** Formulated an optimized data normalization pipeline to maintain high specificity required for real-world medical deployments.
* **Optimized Augmentation:** Implemented custom image augmentation arrays to handle variation in lighting and resolution across clinical datasets.
* **Robust Evaluation:** Utilized precision-recall graphs and confusion matrices to maximize screening throughput while minimizing classification errors.

## Tech Stack & Tools Used
* **Core Frameworks:** Python, PyTorch, Torchvision, CUDA/GPU Acceleration
* **Data & Analytics:** Scikit-Learn Metrics, NumPy, Matplotlib, Google Colab
* **Domain Applications:** Computer Vision, Transfer Learning, Ophthalmology Image Analytics

## Repository Structure
* `diabetic_retinopathy.ipynb`: Executable notebook containing the end-to-end data pipeline, ResNet50 model configuration, and training logs.

## How to Execute the Project
1. Launch the notebook in Google Colab and select the **T4 GPU** environment to ensure accelerated matrix computations.
2. Run the cells sequentially to initialize the preprocessing pipeline, load pretrained weights, and execute the evaluation module.
