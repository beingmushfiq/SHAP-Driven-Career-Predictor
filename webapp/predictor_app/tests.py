from django.test import TestCase, Client
from django.urls import reverse
import os

class TestPredictorViews(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_view(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Predict Your Future")

    def test_predict_get_view(self):
        response = self.client.get(reverse('predict'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Career Assessment Form")

    def test_about_view(self):
        response = self.client.get(reverse('about'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "About the Project")

    def test_analysis_view(self):
        # This might take time to generate plots if they don't exist
        response = self.client.get(reverse('analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Global Feature Importance")
