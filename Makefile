.PHONY: install lint run test clean

VENV := venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip3
CATCOLS := ['Product ID', 'Type']
NUMCOLS := ["UDI",'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TARGETCOL := "Machine failure"
DATECOL := []

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

run_full_training:
	@$(PYTHON) -m	src.Airflow_train_full.py

setup_constants:
	@$(PYTHON) -m	src.CONSTANTS