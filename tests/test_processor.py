import unittest
import pandas as pd
import numpy as np
from src.processor import DataProcessor
from src.config import Config

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.sample_data = pd.DataFrame({
            'logical_quotient': [8, 5],
            'hackathons': [2, 1],
            'coding_skills': [9, 4],
            'public_speaking': [7, 3],
            'self_learning': ['Yes', 'No'],
            'extra_courses': ['No', 'Yes'],
            'certifications': ['Python', 'Machine Learning'],
            'workshops': ['Data Science', 'Web Technologies'],
            'reading_writing_skills': ['Excellent', 'Medium'],
            'memory_capability': ['Excellent', 'Medium'],
            'interested_subjects': ['Programming', 'Mathematics'],
            'interested_career': ['Developer', 'Testing'],
            'company_type': ['Product Development', 'Service Based'],
            'senior_elder_advise': ['Yes', 'No'],
            'book_general_genre': ['Science', 'Fiction'],
            'management_technical': ['Technical', 'Management'],
            'hard_smart_worker': ['Smart Worker', 'Hard Worker'],
            'career_label': ['Software Developer', 'Software Tester']
        })

    def test_clean_data(self):
        cleaned = self.processor.clean_data(self.sample_data)
        self.assertEqual(len(cleaned), 2)
        self.assertIn('logical_quotient', cleaned.columns)

    def test_encode_features(self):
        cleaned = self.processor.clean_data(self.sample_data)
        encoded = self.processor.encode_features(cleaned)
        self.assertTrue(np.issubdtype(encoded['self_learning'].dtype, np.integer))
        self.assertIsNotNone(self.processor.target_encoder)

if __name__ == '__main__':
    unittest.main()
