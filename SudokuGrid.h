#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>

using namespace std;

struct Soln {
    int row;
    int col;
    int val;
};

class SudokuGrid{
public:
	int** grid = NULL;
	int grid_size = 0;
	bool complete = false;

	SudokuGrid(){
	}

	SudokuGrid(int grid_size) {
		this->grid_size = grid_size;
		grid = new int* [grid_size];
		for (int row = 0; row < grid_size; row++) {
			grid[row] = new int[grid_size];
			for (int col = 0; col < grid_size; col++) {
				grid[row][col] = 0;
			}
		}
	}

	~SudokuGrid() {
		for (int row = 0; row < grid_size; row++) {
			delete[] grid[row];
		}
		delete[] grid;
	}

	bool checkComplete(int** grid, int grid_size);
	void solve(int*** grid, int grid_size);
	int* generateRandSoln(int grid_size);
	bool verifySoln(int** sudoku, int val, int row, int col, int grid_size);
	void printGrid();
	void randFirstRow(int*** grid, int grid_size);
	void randClearSlots(int*** grid, int grid_size, int num_zeros, int* soln_row, int* soln_col, int* soln_val);
	int checkZeros();
};

