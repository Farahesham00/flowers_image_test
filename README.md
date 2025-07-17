
# 🌸 Flower Classification with CNN

This project is a deep learning image classification model that distinguishes between five types of flowers:
- Daisy 🌼
- Dandelion 🌾
- Rose 🌹
- Sunflower 🌻
- Tulip 🌷

The model is built using **TensorFlow** and **Keras**, and trained on a dataset of labeled flower images.

---

## 🧠 Project Summary

The main goal of this project is to train a Convolutional Neural Network (CNN) that can classify images of flowers into five distinct categories:

```
Label Mapping:
{
  'daisy': 0,
  'dandelion': 1,
  'rose': 2,
  'sunflower': 3,
  'tulip': 4
}
```

---

## 🧱 Model Architecture

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (128, 128, 3)), 

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'), 
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax'),
])
model.summary()
```

---

## 📊 Model Performance

- **Training Accuracy**: 98.78%
- **Training Loss**: 0.0565  
- **Validation Accuracy**: 60.95%  
- **Validation Loss**: 1.8847

> 📌 Note: The model performs very well on training data, but shows signs of overfitting. Improvements can be made using regularization, dropout, or more data.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/flower-classification-cnn.git
   cd flower-classification-cnn
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Organize your dataset:
   - Place training images inside the `train/` directory.
   - Each subfolder should be named after the flower category:
     ```
     train/
     ├── daisy/
     ├── dandelion/
     ├── rose/
     ├── sunflower/
     └── tulip/
     ```

4. Run the notebook:
   ```bash
   jupyter notebook tutorial.ipynb
   ```

---

## 🗂️ Folder Structure

```
.
├── tutorial.ipynb           # Main notebook
├── train/                   # Training data with subfolders for each class
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── saved_model/             # (Optional) Exported trained model
```

---

## 📦 Requirements

See `requirements.txt`:

```txt
tensorflow>=2.11.0
numpy
matplotlib
pandas
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Farah Hesham**  
AI & ML Engineer | Mechatronics Graduate

---

## 📄 License

This project is licensed under the MIT License.
