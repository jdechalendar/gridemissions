# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DOCS_HOST = $(shell python3 -c "import gridemissions; print(gridemissions.config['DOCS_HOST'])")
ifeq ($(DOCS_HOST),)
$(error DOCS_HOST is not set)
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# https://github.com/sphinx-doc/sphinx/issues/3382#issuecomment-470772316
deploy:
	make html
	rsync -rv --delete _build/html/ $(DOCS_HOST)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
