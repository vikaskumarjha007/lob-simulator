PY ?= python3
export PYTHONPATH := src:.
ES_FILES ?=

.PHONY: setup test demo real

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m lob.demo

# make real ES_FILES="/path/day1/events.parquet /path/day2/events.parquet"   (needs pyarrow)
real:
	$(PY) -m lob.demo --real $(ES_FILES)
