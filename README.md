# ML-Deployment-Series
Experimenting with a series of ML deployment methods.

## 1. Classifier web service
Based on a blog [here](https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf).
A simple classifier, wrapped in a Flask app and containersed by Docker. The Docker is then deployed on a cloud server.
Tech-stack:
- Flask
- Docker
- EC2 deployment

## 2. Classifier web service
- A recommendation system.
Improves based on user input.
- Stores user inputs (Has an associated Data base)
- Re-trains on the inputs
- Has UI

## 3. ML webapp, running on Kubernetes
