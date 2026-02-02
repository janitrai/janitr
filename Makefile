PYTHON ?= python3

prepare:
	$(PYTHON) scripts/prepare_data.py

train: prepare
	$(PYTHON) scripts/train_fasttext.py

eval:
	$(PYTHON) scripts/evaluate.py

all: prepare train eval
