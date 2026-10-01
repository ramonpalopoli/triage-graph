.PHONY: install test lint evals demo
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests evals && ruff format --check src tests evals
evals:
	python evals/run_evals.py $(ARGS)
demo:
	triage-graph "Fui cobrado em dobro, R$$ 320,00, quero reembolso" --offline
