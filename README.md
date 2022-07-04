# Custom docker test

## Setup

Build and run docker container using CLI, UI of cloud provider or VS Code extension.

**Build container**
```
docker build -t custom-docker-test:v0.01 . --platform linux/amd64
```

**Run container**
```
docker run --rm -it --platform linux/amd64 \
    --mount type=bind,source="$(pwd)",target=/custom-docker-test \
    --name custom-docker-test custom-docker-test:v0.01
```

Attach to container using VS Code using SSH connection or VS Code extension.

Q: Do cloud machines support CUDA 10.x and 11.x?
Q: Can we use multiple conda env inside a container?

## Run test

Steps 1, 2, 3 must be done using VS Code inside the container. Step 4 should be run in a server production env and we want use a REST API to inference the model.

1. Go to `notebooks/Explore.ipynb` and run all cells.

2. `python app.py preproc`

Q: How to bring the preprocessing step in the future store?
Q: How to extract data from feature store?

3. `python app.py train`

Q: How to track the experiments?
Q: How to save the model in the model registry?

4. `uvicorn api:app --host 0.0.0.0 --port 80`. Test REST API: `curl -X GET 0.0.0.0:80/predict/ -H 'Content-Type: application' -d @./get.json`

Q: How to bring this application in production (aka kubernetes engine)?


## References

* https://www.kaggle.com/code/swarnabha/pytorch-text-classification-torchtext-lstm/notebook
* https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset
* https://colab.research.google.com/drive/1gX8ERqDMQGTO1fKJwELFO11f66xuKVyP?usp=sharing#scrollTo=O-muOECSWrOt
* https://github.com/rsreetech/PyTorchTextClassificationCustomDataset/blob/main/PyTorchTweetTextClassification.ipynb


