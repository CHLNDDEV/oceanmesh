VERSION=$(shell python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])")

default:
	@echo "\"make publish\"?"

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf build/*
	@rm -rf oceanmesh.egg-info/
	@rm -rf dist/

format:
	isort -rc oceanmesh/ tests/*.py
	black setup.py oceanmesh/ tests/*.py
	blacken-docs README.md
