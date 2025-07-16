import os
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Step 1: Extract dataset from ZIP ===
zip_path = r"C:\Users\RISHITA\Downloads\archive (2).zip"  # <-- zip file path
extract_dir = "data"

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("[INFO] Dataset extracted!")

# === Step 2: Load & preprocess images ===
IMG_SIZE = 64  # Resize images to 64x64
data = []
labels = []

# Update this if the internal folder inside zip is different (e.g. 'train', 'images')
image_dir = os.path.join(extract_dir, "train")
if not os.path.exists(image_dir):
    # fallback: search for any folder with images
    for item in os.listdir(extract_dir):
        potential = os.path.join(extract_dir, item)
        if os.path.isdir(potential):
            image_dir = potential
            break

files = os.listdir(image_dir)[:1000]  # Only load first 1000 images for speed

for img_file in files:
    label = 0 if "cat" in img_file.lower() else 1  # 0 = cat, 1 = dog
    img_path = os.path.join(image_dir, img_file)
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img.flatten())  # flatten 64x64 -> 4096
        labels.append(label)
    except:
        continue

X = np.array(data)
y = np.array(labels)

print(f"[INFO] Loaded {len(X)} images")

# === Step 3: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Train SVM ===
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("[INFO] Model trained!")

# === Step 5: Evaluate ===
y_pred = model.predict(X_test)
print("[INFO] Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === Step 6: Plot sample predictions ===
plt.figure(figsize=(12, 6))
for i in range(6):
    img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
    label = 'Dog' if y_pred[i] == 1 else 'Cat'
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
