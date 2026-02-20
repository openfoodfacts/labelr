image_name := europe-west9-docker.pkg.dev/robotoff/gcf-artifacts/train-yolo:latest

build:
	docker build . -t $(image_name)

push:
	docker push $(image_name)
