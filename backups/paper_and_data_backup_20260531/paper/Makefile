## HistoCorrect paper build script
## Requires: xelatex or pdflatex + bibtex
## Usage: make        -> build PDF
##        make clean  -> remove auxiliary files

MAIN = main
FIGURES = figures/fig1_system_overview.png \
          figures/fig2_feature_tensor.png \
          figures/fig3_java_callchain.png \
          figures/fig4_drift_lifecycle.png

.PHONY: all pdf clean distclean

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(FIGURES)
	tectonic $(MAIN).tex

# Alternative: use pdflatex (drop xeCJK in that case)
pdflatex: $(MAIN).tex references.bib $(FIGURES)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).synctex.gz \
	      $(MAIN).fls $(MAIN).fdb_latexmk

distclean: clean
	rm -f $(MAIN).pdf
