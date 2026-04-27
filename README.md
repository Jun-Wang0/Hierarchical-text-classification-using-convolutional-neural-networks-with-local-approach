# Hierarchical Text Classification using CNN and Local Approach

### Description
This repository contains the implementation of three **Hierarchical Text Classification (HTC)** methods:

* **Level-based Local Classification (LLC)**: Baseline independent level classifiers.
* **LLC with Enhanced Consistency (LLC-EC)**: Cascades parent labels to child levels.
* **CNN with Transfer Learning (CNN-TL-LLC)**: Transfers feature representations (dense layers) from parent models to child models.

**The core objective:** Addressing **"hierarchical inconsistency"** where sub-category predictions conflict with parent categories.

---

### Dataset Information
| Dataset | Levels | Leaf Classes | DOI |
| :--- | :---: | :---: | :--- |
| **DBpedia** | 3 | 219 | [10.3233/SW-140134](https://doi.org/10.3233/SW-140134) |
| **Web of Science (WOS)** | 2 | 134 | [10.17632/9rw3vkcfy4.6](https://doi.org/10.17632/9rw3vkcfy4.6) |

---

### Code Information
The implementation is provided in the following Jupyter Notebooks:
1. `1LLC_CNN_WOS_split.ipynb`: Standard CNN-based LCL.
2. `2LLC_TF_IDF+LR_WOS.ipynb`: Baseline Machine Learning (Logistic Regression).
3. `3LLC-EC_CNN_WOS.ipynb`: Deep Learning with **label-based consistency**.
4. `4LLC-EC_TF-IDF_WOS.ipynb`: ML implementation with label-based consistency.
5. `5CNN-TL-LLC_WOS.ipynb`: **Proposed Transfer Learning** approach for feature fusion.

---

### Methodology
The pipeline follows a structured approach:
1.  **Feature Extraction**: 
    * `TF-IDF` for ML models.
    * `Word Embeddings` (Keras Embedding Layer) for DL models.
2.  **Modeling**: 1D Convolutional Neural Networks (**Conv1D**) with Global Max Pooling.
3.  **Knowledge Transfer**: In `CNN-TL-LLC`, features from the `dense_text_l1` layer are concatenated with Level 2 features before the final classification.

---

### Usage Instructions
1.  **Environment**: Mount Google Drive to access datasets.
2.  **Data Loading**: Load the `.csv` files into the `data` variable.
3.  **Execution**: Run the cells sequentially to perform tokenization, model building, and training.
4.  **Visualization**: Use `plot_history()` to visualize loss and accuracy curves.

---

### Requirements
* `Python 3.x`
* `TensorFlow / Keras`
* `Scikit-learn`
* `Pandas, NumPy`
* `Matplotlib, Seaborn`
