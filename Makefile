default: up

all: run

bash: up
	docker-compose exec app bash

ssh: up
	docker-compose exec app sudo service ssh start

train predict: up
	docker-compose exec app python -m src.$@

toy.test: up
	docker-compose exec app python -m src.modules.model.model

toy.test.trend: up
	docker-compose exec app python -m src.modules.model.model --model=trend

# switch mode
gpu:
	@rm -f Dockerfile docker-compose.yml
	@ln -s docker/docker-compose.gpu.yml docker-compose.yml

cpu:
	@rm -f Dockerfile docker-compose.yml
	@ln -s docker/docker-compose.cpu.yml docker-compose.yml

# run tasks
mlflow-run: up
	docker-compose exec app mlflow run --no-conda .

debug: up
	docker-compose exec app pudb3 encoder.py

# visualization tasks
mlflow-ui: up
	docker-compose exec app mlflow ui --host=0.0.0.0

tensorboard: up
	$(eval logdir:=$(shell ls -trd result/* | tail -n 1))
	echo $(logdir)
	docker-compose exec app tensorboard --host=0.0.0.0 --logdir=$(logdir)

# for docker-compose
up:
	docker-compose up -d

active:
	docker-compose up

ps images down:
	docker-compose $@

im:images

build:
	docker-compose build --no-cache

reup: down up

clean:
	docker-compose down --rmi all
	sudo rm -rf app/__pycache__

clean-result:
	rm -rf result/*
