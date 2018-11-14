.PHONY: init clean lint test
.DEFAULT_GOAL := test

init:
	pip install -r requirements.txt

lint:
	python -m flake8

test: clean lint
	nosetests --with-coverage --cover-erase --cover-html --cover-html-dir=coverage -v tests

clean:
	find . -type f -name '*.pyc' -delete
