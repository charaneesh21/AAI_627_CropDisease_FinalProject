# 🌿 Crop Disease Detection using CNNs (AAI 627 Final Project)

This project builds a deep learning pipeline to classify crop leaf diseases using images from the PlantVillage dataset.  
It is designed as a **production-style system**, using `.py` scripts (not notebooks) and containerized with Docker for full reproducibility.

---

## 🧠 Key Features

- 📦 End-to-end deep learning pipeline: preprocess → train → evaluate  
- 🧱 CNN model built using TensorFlow + Keras  
- 🐳 Fully containerized with Docker for environment consistency  
- 🚫 No notebooks — designed for real-world deployment and portability  

---

## 🗂️ Folder Structure

```
.
├── preprocessing/
│   └── preprocess_data.py      # Resize and normalize images
├── training/
│   └── train_model.py          # CNN training script
├── evaluation/
│   └── evaluate_model.py       # Accuracy/loss evaluation
├── Dockerfile                  # Docker image setup
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignores large files, venv, models, data
```

---

## 📥 Dataset (Not included in repo)

Due to size limits, the dataset is **not included** here.

Please download manually:

👉 [**PlantVillage Dataset (Kaggle)**](https://www.kaggle.com/datasets/emmarex/plantdisease)

Place it inside:

```
Data/PlantVillage/
```

The script will output preprocessed images to:

```
Data/Processed/
```

---

## 🛠️ How to Run (via Docker)

### 1. Build Docker image

```bash
docker build -t crop-disease .
```

### 2. Run container with volume mounts

```bash
docker run -it \
  -v $(pwd)/Data:/app/Data \
  -v $(pwd)/models:/app/models \
  crop-disease
```

### 3. Run inside Docker container

```bash
# Step 1: Preprocess images
python preprocessing/preprocess_data.py

# Step 2: Train the model
python training/train_model.py

# Step 3: Evaluate model performance
python evaluation/evaluate_model.py
```

The trained model will be saved as:

```
models/crop_disease_model.h5
```

---

## 📊 Model Summary

- Architecture: 3-layer CNN  
- Input: 128×128 RGB images  
- Classes: 10 crop disease categories  
- Accuracy: ~92% on validation set  
- Loss Function: Categorical Crossentropy  
- Optimizer: Adam  

---

## ✅ Requirements

Installed automatically inside Docker:

```
tensorflow
opencv-python
numpy
```

---

- Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🧪 Future Work

- Deploy model via Gradio or Streamlit  
- Train on all 38 classes of PlantVillage dataset  
- Add data augmentation and learning rate scheduling

---
