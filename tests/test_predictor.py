import unittest
import numpy as np
from src.predictor import CareerPredictor
from src.config import Config

class TestCareerPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We assume training has been run and artifacts exist
        cls.predictor = CareerPredictor.get_instance()

    def test_singleton(self):
        another_instance = CareerPredictor.get_instance()
        self.assertIs(self.predictor, another_instance)

    def test_preprocess_input(self):
        sample_input = {
            'logical_quotient': 8,
            'hackathons': 2,
            'coding_skills': 9,
            'public_speaking': 7,
            'self_learning': 'Yes',
            'extra_courses': 'No',
            'certifications': 'Python',
            'workshops': 'Data Science',
            'reading_writing_skills': 'Excellent',
            'memory_capability': 'Excellent',
            'interested_subjects': 'Programming',
            'interested_career': 'Developer',
            'company_type': 'Product Development',
            'senior_elder_advise': 'Yes',
            'book_general_genre': 'Science',
            'management_technical': 'Technical',
            'hard_smart_worker': 'Smart Worker'
        }
        encoded = self.predictor.preprocess_input(sample_input)
        self.assertEqual(encoded.shape, (1, 17))

    def test_predict(self):
        sample_input = {
            'logical_quotient': 5,
            'hackathons': 1,
            'coding_skills': 5,
            'public_speaking': 5,
            'self_learning': 'No',
            'extra_courses': 'Yes',
            'certifications': 'Machine Learning',
            'workshops': 'Web Technologies',
            'reading_writing_skills': 'Medium',
            'memory_capability': 'Medium',
            'interested_subjects': 'Mathematics',
            'interested_career': 'Testing',
            'company_type': 'Service Based',
            'senior_elder_advise': 'No',
            'book_general_genre': 'Fiction',
            'management_technical': 'Management',
            'hard_smart_worker': 'Hard Worker'
        }
        label, probs = self.predictor.predict(sample_input)
        self.assertIsInstance(label, str)
        self.assertEqual(len(probs), len(self.predictor.get_class_names()))

if __name__ == '__main__':
    unittest.main()
