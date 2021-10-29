#!/bin/bash -e
pylint --rcfile=./.pylintrc autogbm | grep unused-import
pylint --rcfile=./.pylintrc autogbm | grep missing-function-docstring
pylint --rcfile=./.pylintrc autogbm | grep unused-variable
pylint --rcfile=./.pylintrc autogbm | grep import-outside-toplevel
