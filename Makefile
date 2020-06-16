PYTHON?=python3
PYTHONPATH?=./
SOURCE_DIR=./lightfm
TESTS_DIR=./tests
PEP8=pep8

.PHONY: examples
examples:
	jupyter nbconvert --to rst examples/quickstart/quickstart.ipynb
	mv examples/quickstart/quickstart.rst doc/
	jupyter nbconvert --to rst examples/movielens/example.ipynb
	mv examples/movielens/example.rst doc/examples/movielens_implicit.rst
	jupyter nbconvert --to rst examples/movielens/learning_schedules.ipynb
	mv examples/movielens/learning_schedules.rst doc/examples/
	cp -r examples/movielens/learning_schedules_files doc/examples/
	rm -rf examples/movielens/learning_schedules_files
	jupyter nbconvert --to rst examples/stackexchange/hybrid_crossvalidated.ipynb
	mv examples/stackexchange/hybrid_crossvalidated.rst doc/examples/
	jupyter nbconvert --to rst examples/movielens/warp_loss.ipynb
	mv examples/movielens/warp_loss.rst doc/examples/
	cp -r examples/movielens/warp_loss_files doc/examples/
	rm -rf examples/movielens/warp_loss_files

.PHONY: update-docs
update-docs:
	pip install -e . \
	&& cd doc && make html && cd .. \
	&& git fetch origin gh-pages && git checkout gh-pages \
	&& rm -rf ./docs/ \
	&& mkdir ./docs/ \
	&& cp -r ./doc/_build/html/* ./docs/ \
	&& git add -A ./docs/* \
	&& git commit -m 'Update docs.' && git push origin gh-pages

clean: clean-pyc clean-test

clean-pyc:
	find . -name '*.py[cod]' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*$py.class' -exec rm -rf {} +

clean-test:
	rm -rf $(COVERAGE_HTML_REPORT_DIR)

test: compile
	py.test -v -m 'not long' tests

check: pep8

pep8:
	$(PEP8) $(SOURCE_DIR) $(TESTS_DIR) $(BIN_DIR)

compile:
	python setup.py cythonize
	python setup.py build_ext --inplace

bump:
	@bumpversion --commit --tag patch

coverage:
	@coverage run --source lightfm -m py.test
	@coverage report
