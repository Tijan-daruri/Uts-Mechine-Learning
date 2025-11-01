# Laporan Singkat: Klasifikasi Dataset Iris

**Deskripsi dataset**
Dataset Iris (150 sampel, 4 fitur numerik: sepal length, sepal width, petal length, petal width) dan 3 kelas target (setosa, versicolor, virginica). Data diambil dari scikit-learn.

**Preprocessing & EDA**
- Pembagian data: train 75% / test 25% (stratified).
- Scaling: StandardScaler untuk model yang memerlukan scaling (Logistic Regression).
- Distribusi kelas: {'setosa': 50, 'versicolor': 50, 'virginica': 50}

**Model yang digunakan**
1. Logistic Regression (one-vs-rest) — menggunakan fitur terstandardisasi.
2. Decision Tree (depth auto) — menggunakan fitur asli (trees tidak memerlukan scaling).

**Hasil evaluasi (test set)**
| Model              |   Accuracy |   Precision_macro |   Recall_macro |   F1_macro |
|:-------------------|-----------:|------------------:|---------------:|-----------:|
| LogisticRegression |   0.815789 |          0.827778 |       0.818376 |   0.820745 |
| DecisionTree       |   0.894737 |          0.90303  |       0.897436 |   0.896825 |

Laporan singkat ini menyertakan confusion matrix dan ROC plots yang disimpan di folder `classification_project_iris`.
Gambar confusion matrix dan ROC untuk masing-masing model:
- Confusion matrix LogisticRegression: confusion_matrix_LogisticRegression.png
- ROC LogisticRegression: roc_LogisticRegression.png
- Confusion matrix DecisionTree: confusion_matrix_DecisionTree.png
- ROC DecisionTree: roc_DecisionTree.png

**Pembahasan dan Kesimpulan**
- Kedua model menunjukkan akurasi tinggi pada dataset Iris (dataset bersih dan linear-separable untuk beberapa kelas).
- Logistic Regression bekerja sangat baik pada kelas setosa — garis pemisah linear cukup untuk memisahkan sebagian kelas. Decision Tree juga memberikan hasil kuat dan kadang lebih mudah diinterpretasi (dapat mengekstrak aturan).
- Untuk tugas nyata, perlu cross-validation, tuning hyperparameter (grid search), dan pemeriksaan overfitting (learning curves).
- Rekomendasi: jika interpretabilitas penting, gunakan Decision Tree; jika kestabilan dan probabilitas prediksi diperlukan, Logistic Regression.

---
**File yang disertakan**
- `classification_project_iris.py` — script Python lengkap untuk menjalankan analisis.
- Gambar dan file CSV hasil EDA dan evaluasi di folder `classification_project_iris`.
