# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0

.PHONY: install test unit-test eval lint playground deploy

install:
	uv sync

unit-test:
	uv run pytest tests/unit -v

test:
	uv run pytest tests/unit -v

eval:
	agents-cli eval app \
		--eval-dataset-path tests/eval/datasets/basic-dataset.json \
		--config-path tests/eval/eval_config.yaml \
		--output-dir eval_results

lint:
	uv run ruff check app tests

playground:
	agents-cli playground

deploy:
	agents-cli deploy --target cloud_run
