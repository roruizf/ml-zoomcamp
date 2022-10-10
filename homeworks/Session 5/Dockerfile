# Base image
FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv
#It creates a new "app" directory in the image once run
WORKDIR /app
#files to copy int he image and the destination directory
COPY ["Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy

COPY ["webservice_docker.py", "./"] 

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "webservice_docker:app" ]

# Build the image from Dockerfile
# docker build -t zoomcamp-s5 .

#Run the just built image
# docker run -it --rm -p 9696:9696 zoomcamp-s5