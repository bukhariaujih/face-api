from locust import HttpUser, TaskSet, task, between, tag

"""
Run locus with:
locust -f ./tests/load_test.py
"""


class FacePredict(TaskSet):
    @tag('Predictions')
    @task
    def predict(self):
        request_body = {"data": [[4.8, 3, 1.4, 0.3]]}
        self.client.post('/v1/face/predict', json=request_body)

    @tag('Baseline')
    @task
    def health_check(self):
        self.client.get('/')


class FaceLoadTest(HttpUser):
    tasks = [FacePredict]
    host = 'http://127.0.0.1'
    stop_timeout = 200
    wait_time = between(1, 5)