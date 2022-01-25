.PHONY: help build train
.DEFAULT_GOAL := help

NOW = $(shell date '+%Y%m%d-%H%M%S-%N')


train: ## Run training
	@nohup python train.py +settings.run_fold=0 > /tmp/nohup_$(NOW).log &
	sleep 2
	@nohup python train.py +settings.run_fold=1 > /tmp/nohup_$(NOW).log &
	sleep 2
	@nohup python train.py +settings.run_fold=2 > /tmp/nohup_$(NOW).log &
	sleep 2
	@nohup python train.py +settings.run_fold=3 > /tmp/nohup_$(NOW).log &
	sleep 2
	@nohup python train.py +settings.run_fold=4 > /tmp/nohup_$(NOW).log &
	sleep 2
	@nohup python train.py +settings.run_fold=5 > /tmp/nohup_$(NOW).log &
	sleep 2
	@nohup python train.py +settings.run_fold=6 > /tmp/nohup_$(NOW).log &

debug: ## Run training debug mode
	@python train.py settings.debug=True hydra.verbose=True +settings.run_fold=1

build: clean-build ## Build package
	@python encode.py ./src ./config

clean: clean-build clean-pyc clean-training ## Remove all build and python artifacts

clean-build: ## Remove build artifacts
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove python artifacts
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-training: ## Remove training artifacts
	@rm -rf ../outputs ../multirun

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
