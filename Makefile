test:
	pytest tests/ --cov=policyengine_data --cov-report=xml --maxfail=0 -v

install:
	pip install -e ".[dev]"

check-format:
	linecheck .
	isort --check-only --profile black src/
	black . -l 79 --check

format:
	linecheck . --fix
	isort --profile black src/
	black . -l 79

documentation:
	cd docs && jupyter-book build .
	python docs/add_plotly_to_book.py docs/_build/html

build:
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf docs/_build/

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.1.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine_data --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml
