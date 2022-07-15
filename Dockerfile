FROM quay.io/thoth-station/s2i-minimal-f34-py39-notebook:v0.3.0

LABEL name="s2i-pytorch-nltk-notebook:latest" \
      summary="PyTorch NLTK Jupyter Notebook Source-to-Image for Python 3.9 applications." \
      description="Notebook image based on Source-to-Image.These images can be used in OpenDatahub JupterHub." \
      io.k8s.description="Notebook image based on Source-to-Image.These images can be used in OpenDatahub JupterHub." \
      io.k8s.display-name="PyTorch NLTK Notebook Python 3.9-ubi8 S2I" \
      io.openshift.expose-services="8080:http" \
      io.openshift.tags="python,python39" \
      authoritative-source-url="https://quay.io/mmurakam/s2i-pytorch-nltk-notebook" \
      io.openshift.s2i.build.commit.ref="main" \
      io.openshift.s2i.build.source-location="https://github.com/mamurak/custom-docker-test" \
      io.openshift.s2i.build.image="quay.io/thoth-station/s2i-minimal-f34-py39-notebook:v0.3.0"

USER root
WORKDIR /tmp/

# Copying custom packages
COPY requirements.txt /tmp/

# Install custom packages
RUN micropipenv install --deploy 

WORKDIR /opt/app-root/src
USER 1001

CMD /opt/app-root/bin/start-singleuser.sh --ip=0.0.0.0 --port=8080
