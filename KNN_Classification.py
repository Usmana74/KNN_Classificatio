import numpy as np
import pandas as pd
from collections import Counter

dataset=pd.read_csv('acid_dataset.csv')

def knn_classification(dataset,newacid,k=3):

  features=dataset[['Acidity','Odour']].values
  labels = dataset['Label'].values

  distance=np.linalg.norm(features-newacid, axis=1)

  k_indices=np.argsort(distance)[:k]

  k_labels= labels[k_indices]

  most_common= Counter(k_labels).most_common(1)

  return most_common[0][0]


newacid=np.array([4.37,2.4])

predicted_label= knn_classification(dataset, newacid, k=3)

print(f'The predicted label for new acid is:{predicted_label}')