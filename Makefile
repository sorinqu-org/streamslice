PYTHON ?= python3

.PHONY: doctor test typecheck process-test

doctor:
	PYTHONPATH=src $(PYTHON) -m streamslice.cli --config config/default.yaml doctor

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

typecheck:
	cd remotion && npm run typecheck

process-test:
	PYTHONPATH=src $(PYTHON) -m streamslice.cli --config config/test.yaml process \
		--input /home/yuwye/streams/t2x2/2026-07-25_14-09-05/chunk_4.mp4

