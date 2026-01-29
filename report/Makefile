report.pdf: report.tex dgemm_seq.tex dgemm_omp.tex dgemm_mpi.tex dgemm_cuda.tex appendixes_cuda.tex references.bib
	pdflatex report.tex
	biber report.bcf
	pdflatex report.tex

clean:
	rm -f report.aux
	rm -f report.bbl
	rm -f report.bcf
	rm -f report.blg
	rm -f report.log
	rm -f report.nav
	rm -f report.out
	rm -f report.pdf
	rm -f report.snm
	rm -f report.toc
	rm -f report.run.xml
