PYTHON ?= python3

.PHONY: help test phase1 all clean

help:
	@echo "make test    run the test suite"
	@echo "make phase1  build the behavioural target and Phase 1 reports"
	@echo "make phase2  fit and validate the behavioural scorecard"
	@echo "make phase3  uplift, Qini and exposure impact"
	@echo "make phase4  decision bands and champion against challenger"
	@echo "make all     run every implemented phase"
	@echo "make clean   remove cached interim and processed data"

test:
	$(PYTHON) -m pytest -q

phase1:
	$(PYTHON) run.py 1

phase2:
	$(PYTHON) run.py 2

phase3:
	$(PYTHON) run.py 3

phase4:
	$(PYTHON) run.py 4

all:
	$(PYTHON) run.py all

clean:
	rm -rf data/interim data/processed
	find . -name __pycache__ -type d -exec rm -rf {} +
