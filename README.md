# PRODIGY_ML_03
A Support Vector Machine (SVM) classifier that identifies cats and dogs using grayscale image data from the Kaggle dataset. The script extracts images from a ZIP file, preprocesses them, trains the model, and visualizes predictions.
# 🐱🐶 Cat vs Dog Image Classifier with SVM

This project uses a **Support Vector Machine (SVM)** to classify images of cats and dogs using grayscale image data. It includes a dataset shrinking tool, training script, and visualization of predictions.

---

## 🧠 Tech Stack

- Python 🐍
- OpenCV (image processing)
- Scikit-learn (SVM & ML tools)
- Matplotlib (for plotting results)

---

## 📦 Dataset

This project uses the [Dogs vs. Cats dataset from Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).

### 🔄 Shrinking the Dataset

To keep the project lightweight (for GitHub upload or testing), use the included script to extract only a portion of the full dataset.

### ➕ Script: `shrink_dataset.py`

```bash
python shrink_dataset.py
