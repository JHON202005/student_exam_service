import inspect
from django.test import TestCase
from apps.ml.exam_classifier.random_forest import RandomForestClassifier
from apps.ml.registry import MLRegistry

class MLTests(TestCase):

    def test_rf_algorithm(self):
        input_data = {
            "gender": "male",
            "race/ethnicity": "group B",
            "parental level of education": "bachelor's degree",
            "lunch": "standard",
            "test preparation course": "none",
            "reading score": 72,
            "writing score": 74
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertIn(response['label'], ['Pass', 'Fail'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)

        endpoint_name = "exam_score_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "YourName"
        algorithm_description = "Random Forest for student exam pass/fail prediction"
        algorithm_code = inspect.getsource(RandomForestClassifier)

        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                               algorithm_status, algorithm_version, algorithm_owner,
                               algorithm_description, algorithm_code)

        self.assertEqual(len(registry.endpoints), 1)