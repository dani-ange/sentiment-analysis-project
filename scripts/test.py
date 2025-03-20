import unittest
from transformers import pipeline

classifier = pipeline("text-classification", model="./models")

class TestModel(unittest.TestCase):
    def test_positive_sentiment(self):
        result = classifier("I love this product!")[0]
        self.assertIn(result["label"], ["LABEL_0", "LABEL_1", "LABEL_2"])

    def test_negative_sentiment(self):
        result = classifier("This is terrible, I hate it.")[0]
        self.assertIn(result["label"], ["LABEL_0", "LABEL_1", "LABEL_2"])

if __name__ == "__main__":
    unittest.main()
