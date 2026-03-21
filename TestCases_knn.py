import pickle
import unittest

from KNN import __authors__, __group__, KNN
from utils import *
from utils_data import read_dataset


class TestCases(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        with open('./test/test_cases_knn.pkl', 'rb') as f:
            self.test_cases = pickle.load(f)

    def test_NIU(self):
        # DON'T FORGET TO WRITE YOUR NIU AND GROUPS
        self.assertNotEqual(__authors__, "Hugo Aguilar", msg="1633543")
        self.assertNotEqual(__group__, "73", msg="73")
        self.assertIsInstance(__authors__, list)
        for author in __authors__:
            self.assertIsInstance(author, str)
            self.assertEqual(len(author), 7)
            self.assertTrue(author.isnumeric())

    def test_init_train(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            np.testing.assert_array_equal(
                knn.train_data, self.test_cases['init_train'][ix])

    def test_get_k_neighbours(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            knn.get_k_neighbours(
                self.test_cases['test_input'][ix][0], self.test_cases['rnd_K'][ix])
            np.testing.assert_array_equal(
                knn.neighbors, self.test_cases['get_k_neig'][ix])

    def test_get_class(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            knn.get_k_neighbours(
                self.test_cases['test_input'][ix][0], self.test_cases['rnd_K'][ix])
            preds = knn.get_class()
            np.testing.assert_array_equal(
                preds, self.test_cases['get_class'][ix])

    def test_fit(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            preds = knn.predict(
                self.test_cases['test_input'][ix][0], self.test_cases['rnd_K'][ix])
            np.testing.assert_array_equal(
                preds, self.test_cases['get_class'][ix])


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

result = read_dataset(
    root_folder='./images/', 
    gt_json='./images/gt.json', 
    with_color=False
)

train_imgs   = result[0]
train_labels = result[1]
test_imgs    = result[3]
test_labels  = result[4]

print("Q4 - Num train images:", len(train_imgs))

knn = KNN(train_imgs, train_labels)

print("Q5 (K=5):", knn.predict(test_imgs[0:1], k=5)[0])
print("Q6 (K=2):", knn.predict(test_imgs[0:1], k=2)[0])

knn = KNN(train_imgs, train_labels)

# Predecir las primeras 20 imágenes del test con K=5 y K=7
for i in range(20):
    p5 = knn.predict(test_imgs[i:i+1], k=5)[0]
    p7 = knn.predict(test_imgs[i:i+1], k=7)[0]
    print(f"Test[{i}] - K=5: {p5}, K=7: {p7}, real: {test_labels[i]}")

for i in range(20, 50):
    p5 = knn.predict(test_imgs[i:i+1], k=5)[0]
    p7 = knn.predict(test_imgs[i:i+1], k=7)[0]
    print(f"Test[{i}] - K=5: {p5}, K=7: {p7}, real: {test_labels[i]}")

import json
with open('./images/gt.json') as f:
    gt = json.load(f)

test_ids = list(gt['test'].keys())

import matplotlib.pyplot as plt

result = read_dataset(
    root_folder='./images/', 
    gt_json='./images/gt.json', 
    with_color=False
)

from PIL import Image
import numpy as np

# Cargar y preprocesar las imágenes
img1 = Image.open('./images/test/Imatge1.png').convert('L')
img1 = img1.resize((60, 80))
img1_array = np.array(img1).reshape(1, 80, 60)

img2 = Image.open('./images/test/imatge2.png').convert('L')
img2 = img2.resize((60, 80))
img2_array = np.array(img2).reshape(1, 80, 60)

# Pregunta 7: img1 con K=5
print("P7 (img1, K=5):", knn.predict(img1_array, k=5)[0])

# Pregunta 8: img1 con K=7
print("P8 (img2, K=7):", knn.predict(img2_array, k=7)[0])

# Pregunta 9: img2 con K=5
print("P9 (img2, K=5):", knn.predict(img2_array, k=5)[0])
