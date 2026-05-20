from django.test import TestCase
from rest_framework.test import APIClient

class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
        input_data = {
            "gender": "male",
            "race/ethnicity": "group B",
            "parental level of education": "bachelor's degree",
            "lunch": "standard",
            "test preparation course": "none",
            "reading score": 72,
            "writing score": 74
        }
        classifier_url = "/api/v1/exam_score_classifier/predict"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.data["label"], ["Pass", "Fail"])
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)