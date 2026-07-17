SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: help docs docs-open examples

help: ## Show documentation targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Mirtal commands:\n\n"} \
		/^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

docs: ## Build rustdoc for the complete public workspace API.
	@RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

docs-open: ## Build and open the mirtal API documentation.
	@RUSTDOCFLAGS="-D warnings" cargo doc -p mirtal --no-deps --open

examples: ## Type-check every public mirtal example.
	@cargo check -p mirtal --examples
