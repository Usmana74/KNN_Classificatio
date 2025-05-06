# KNN Classification for Acid Dataset

This project implements a simple **K-Nearest Neighbors (KNN)** algorithm in Python using NumPy and Pandas. It classifies a new acid sample based on its *Acidity* and *Odour* values using the provided dataset.

## 📁 Dataset

The `acid_dataset.csv` file contains data with the following columns:

- `Acidity` (float): A numeric value representing the acidity level of a sample.
- `Odour` (float): A numeric representation of the odour intensity or property.
- `Label` (string/int): The class label corresponding to the sample (e.g., "Strong", "Mild", "Neutral").

##  How It Works

1. **Feature Extraction**: Extracts Acidity and Odour as feature vectors.
2. **Distance Calculation**: Computes the Euclidean distance from the new data point to all points in the dataset.
3. **KNN Logic**: Identifies the `k` nearest neighbors and performs a majority vote using the labels.
4. **Prediction**: Outputs the predicted label for the new input sample.

##  Dependencies

- `numpy`
- `pandas`
- `collections` (standard library)

Install dependencies using:

```bash
pip install numpy pandas

