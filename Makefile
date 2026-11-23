.PHONY: install test lint evals demo api docker
install:
	pip install -e ".[dev,api]"
test:
	pytest -q
lint:
	ruff check src tests evals && ruff format --check src tests evals
evals:
	python evals/run_evals.py $(ARGS)
demo:
	triage-graph "Fui cobrado em dobro, R$$ 320,00, quero reembolso" --offline
api:
	uvicorn --factory triage_graph.api:create_app --reload
docker:
	docker build -t triage-graph .
