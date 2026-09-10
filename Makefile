PYTHON ?= python3

.PHONY: doctor test typecheck process-test

doctor:
	PYTHONPATH=src $(PYTHON) -m streamslice.cli --config config/default.yaml doctor

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

typecheck:
	cd remotion && npm run typecheck

# make process-test INPUT=/path/to/chunk.mp4
INPUT ?=

process-test:
	@test -n "$(INPUT)" || { echo "usage: make process-test INPUT=/path/to/chunk.mp4"; exit 2; }
	PYTHONPATH=src $(PYTHON) -m streamslice.cli --config config/test.yaml process --input $(INPUT)

