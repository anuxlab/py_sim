.PHONY: install test lint demo trace-demo benchmark ci

install:
	pip install -r requirements-dev.txt -r requirements-analysis.txt

test:
	python3 -m pytest -v --cov=k8s_sim --cov-report=term-missing

lint:
	flake8 k8s_sim tests --select=E9,F63,F7,F82 --show-source
	flake8 k8s_sim tests --max-line-length=120 --exit-zero

demo:
	python3 demo.py

trace-demo:
	python3 trace_demo.py openb_pod_list_gpushare100.csv 400

benchmark:
	python3 experiments/run_benchmark.py --preset fast

# Runs everything CI runs, locally.
ci: lint test demo trace-demo
