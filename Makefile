.PHONY: setup format lint sync-devtools help

setup format lint sync-devtools help:
	git submodule update --init --recursive
	python .devtools/scripts/devtools.py $@ --langs $(LANGS)
