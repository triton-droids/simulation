.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := build

.PHONY: clean
clean:
	@rm -rf logs/

.PHONY: wandb
wandb:
	vcs import < tools/repos/wandb.repos
	source tools/scripts/wandb.sh
