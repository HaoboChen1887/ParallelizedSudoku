compile: main.cu
	nvcc -rdc=true -o main SudokuGrid.cu SolutionGrid.cu main.cu -std=c++11
