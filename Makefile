.PHONY: install lint run test clean

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install: $(VENV)

$(VENV): requirements.txt
	
	@$(PIP) install -r requirements.txt

format:
	@$(VENV)/bin/black my_project

lint:
	@$(VENV)/bin/flake8 my_project

run:
	@$(PYTHON) my_project/main.py

test:
	@$(PYTHON) -m unittest discover -s tests -p '*_test.py'

clean:
	@rm -rf $(VENV)

hello:
	@echo "Hello World"