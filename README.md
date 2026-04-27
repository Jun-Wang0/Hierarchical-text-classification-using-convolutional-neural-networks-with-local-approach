Hierarchical Text Classification using CNN and Local Approach
Description
This repository contains the implementation of three hierarchical text classification (HTC) methods:

Level-based Local Classification (LLC): Baseline independent level classifiers.

LLC with Enhanced Consistency (LLC-EC): Cascades parent labels to child levels.

CNN with Transfer Learning (CNN-TL-LLC): Transfers feature representations (dense layers) from parent models to child models.

The project addresses "hierarchical inconsistency" where sub-category predictions conflict with parent categories.

Dataset Information
The models are evaluated on two benchmark datasets:

DBpedia: Wikipedia-based hierarchy (3 levels, 219 leaf classes). doi:10.3233/SW-140134

Web of Science (WOS): Academic abstracts (2 levels, 134 leaf classes). doi: 10.17632/9rw3vkcfy4.6

Code Information
1LLC_CNN_WOS_split.ipynb: Standard CNN-based LCL implementation.

2LLC_TF_IDF+LR_WOS.ipynb: Baseline Machine Learning (Logistic Regression) implementation.

3LLC-EC_CNN_WOS.ipynb: Deep Learning implementation with label-based consistency.

4LLC-EC_TF-IDF_WOS.ipynb: Machine Learning implementation with label-based consistency.

5CNN-TL-LLC_WOS.ipynb: The proposed Transfer Learning approach for feature fusion.

Usage Instructions
Mount Google Drive to access datasets.

Load the .csv files (e.g., wos_train_final.csv) into the data variable.

Run the cells sequentially to perform tokenization, model building, and training.

Use the plot_history function to visualize loss and accuracy curves.

Requirements
Python 3.x

TensorFlow / Keras

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

Methodology
The pipeline follows:

Feature Extraction: TF-IDF for ML models; Word Embeddings (Keras Embedding Layer) for DL models.

Modeling: 1D Convolutional Neural Networks (Conv1D) with Global Max Pooling.

Knowledge Transfer: In CNN-TL-LLC, features from the dense_text_l1 layer are concatenated with Level 2 features before the final classification.
