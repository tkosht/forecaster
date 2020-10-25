default: up

all: run

bash: up
	docker-compose exec app bash

# switch mode
gpu:
	@rm -f Dockerfile docker-compose.yml
	@ln -s docker/docker-compose.gpu.yml docker-compose.yml

cpu:
	@rm -f Dockerfile docker-compose.yml
	@ln -s docker/docker-compose.cpu.yml docker-compose.yml

kedro: up
	docker-compose exec app kedro new --config kedro.yml

kedro-viz: up
	docker-compose exec --workdir=/home/dsuser/workspace/kedro app kedro viz --host=0.0.0.0

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
