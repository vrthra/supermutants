all:
	python3 supermutant.py callgraph.csv mutants.csv

debug:
	pudb.sh supermutant.py callgraph.csv mutants.csv
