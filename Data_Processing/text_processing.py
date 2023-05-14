from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

class TextToNumeric:
    def __init__(self, df, text_columns):
        self.df = df
        self.text_columns = text_columns

    def transform_with_tfidf(self):
        for col in self.text_columns:
            vectorizer = TfidfVectorizer()
            transformed_data = vectorizer.fit_transform(self.df[col])
            tfidf_df = pd.DataFrame(transformed_data.toarray(), 
                                    columns=vectorizer.get_feature_names_out())
            # Drop the original text column
            self.df = self.df.drop(columns=[col])
            # Join the new dataframe with the original dataframe
            self.df = pd.concat([self.df, tfidf_df], axis=1)
        return self.df

    def transform_with_count_vectorizer(self):
        for col in self.text_columns:
            vectorizer = CountVectorizer()
            transformed_data = vectorizer.fit_transform(self.df[col])
            count_vect_df = pd.DataFrame(transformed_data.toarray(), 
                                         columns=vectorizer.get_feature_names_out())
            # Drop the original text column
            self.df = self.df.drop(columns=[col])
            # Join the new dataframe with the original dataframe
            self.df = pd.concat([self.df, count_vect_df], axis=1)
        return self.df

import unittest
import pandas as pd
import numpy as np

class TestTextToNumeric(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'text': ['This is a sentence', 'This is another sentence', 'This is yet another sentence'],
            'target': [1, 0, 1]
        })
        self.text_to_numeric = TextToNumeric(self.df, ['text'])

    def test_transform_with_tfidf(self):
        transformed_df = self.text_to_numeric.transform_with_tfidf()
        self.assertEqual(set(transformed_df.columns), set(['target', 'this', 'is', 'a', 'sentence', 'another', 'yet']))

    def test_transform_with_count_vectorizer(self):
        transformed_df = self.text_to_numeric.transform_with_count_vectorizer()
        self.assertEqual(set(transformed_df.columns), set(['target', 'this', 'is', 'a', 'sentence', 'another', 'yet']))

if __name__ == "__main__":
    unittest.main()