#include "SudokuGrid.h"

using namespace std;

bool SudokuGrid::checkComplete(int** grid, int grid_size) {
	bool grid_complete = true;
	for (int i = 0; i < grid_size; i++) {
		for (int j = 0; j < grid_size; j++) {
			if (grid[i][j] == 0) {
				grid_complete = false;
				return grid_complete;
			}
		}
	}
	return grid_complete;
}

void SudokuGrid::randFirstRow(int*** grid, int grid_size){
//	srand(time(0));
	int** sudoku = *grid;
    int row = 0;
    for (int col = 0; col < grid_size; col++) {
        int * rand_soln = generateRandSoln(grid_size);
        for (int idx = 0; idx < grid_size; idx++) {
            if (verifySoln(sudoku, rand_soln[idx], row, col, grid_size)) {
                sudoku[row][col] = rand_soln[idx];
                delete[] rand_soln;
                break;
            }
        }
	}

}

void SudokuGrid::solve(int*** grid, int grid_size) {
	int** sudoku = *grid;
    vector<Soln> solution;
	for (int row = 0; row < grid_size; row++) {
		for (int col = 0; col < grid_size; col++) {
			//printGrid();
			bool soln_found = false;
			int soln_val = 0;
			if (sudoku[row][col] == 0) {
				for (soln_val = 1; soln_val < grid_size + 1; soln_val++) {
					if (verifySoln(sudoku, soln_val, row, col, grid_size)) {
						Soln soln_struc;
						soln_struc.row = row;
						soln_struc.col = col;
						soln_struc.val = soln_val;
						sudoku[row][col] = soln_val;
						solution.push_back(soln_struc);
						soln_found = true;
						break;
					}
				}

				//printGrid();
				while (!soln_found) {
					Soln curr = solution.back();
					solution.pop_back();
					row = curr.row;
					col = curr.col;
					sudoku[row][col] = 0; // reset value
					//printGrid();
					for (soln_val = curr.val + 1; soln_val < grid_size + 1; soln_val++) {
						if (verifySoln(sudoku, soln_val, row, col, grid_size)) {
							Soln soln_struc;
							soln_struc.row = row;
							soln_struc.col = col;
							soln_struc.val = soln_val;
							sudoku[row][col] = soln_val;
							solution.push_back(soln_struc);
							soln_found = true;
							break;
							//printGrid();
						}
					}
				}
			}
		}
	}
}

void SudokuGrid::randClearSlots(int*** grid, int grid_size, int num_zeros, int* soln_row, int* soln_col, int* soln_val) {
//	srand(time(0));
	int** sudoku = *grid;
	int row = rand() % grid_size;
	int col = rand() % grid_size;
	for (int cnt = 0; cnt < num_zeros; cnt++) {
		while (sudoku[row][col] == 0) {
			row = rand() % grid_size;
			col = rand() % grid_size;
		}
		sudoku[row][col] = 0;
		soln_row[cnt] = row;
		soln_col[cnt] = col;
		soln_val[cnt] = 0;
	}
}

int* SudokuGrid::generateRandSoln(int grid_size) {
	int* soln = new int[grid_size];
	for (int idx = 0; idx < grid_size; idx++) {
		int val = 0;
		do {
			val = rand() % grid_size + 1;
		} while(find(soln, soln + grid_size, val) != soln + grid_size);
		soln[idx] = val;
	}
	return soln;
}

bool SudokuGrid::verifySoln(int** sudoku, int val, int row, int col, int grid_size) {
	bool is_valid = false;
	for (int idx = 0; idx < grid_size; idx++) {
		if (sudoku[row][idx] == val)
			return is_valid;
		if (sudoku[idx][col] == val)
			return is_valid;
	}
	int sub_size = (int)sqrt(grid_size);
	int sub_row = row / sub_size;
	int sub_col = col / sub_size;
	int g_row = sub_row * sub_size;
	int g_col = sub_col * sub_size;
	for (int i = g_row; i < g_row + sub_size; i++) {
		for (int j = g_col; j < g_col + sub_size; j++) {
			if (sudoku[i][j] == val) {
				return is_valid;
			}
		}
	}

//	cout << "-----" << endl;
//	for (int i = g_row; i < g_row + sub_size; i++) {
//		for (int j = g_col; j < g_col + sub_size; j++) {
//			cout << sudoku[i][j] << " ";
//		}
//		cout << "|" << endl;
//	}
//	cout << "-----" << endl;
	is_valid = true;
	return is_valid;
}

void SudokuGrid::printGrid() {
	cout << "=======================" << endl;
	for (int row = 0; row < grid_size; row++) {
		int sub_size = (int)sqrt(grid_size);
		for (int col = 0; col < grid_size; col++) {
			if (col % sub_size == 0 && col != 0)
				cout << "| ";
			cout << grid[row][col] << " ";
		}
		cout << endl;
		if ((row + 1) % sub_size == 0) {
			cout << "------------------------" << endl;
		}
	}
}

int SudokuGrid::checkZeros() {
	int result = 0;
	for (int row = 0; row < grid_size; row++) {
		for (int col = 0; col < grid_size; col++) {
			if (grid[row][col] == 0)
				result++;
		}
	}
	return result;
}
