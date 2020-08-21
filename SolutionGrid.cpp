#include "SolutionGrid.h"

__host__ __device__ bool checkComplete(int** grid, int grid_size) {
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

int* sudokuInit(int* grid, int** config, int grid_size) {
	grid = new int [grid_size * grid_size];
	for (int row = 0; row < grid_size; row++) {
		for (int col = 0; col < grid_size; col++) {
			grid[row * grid_size + col] = config[row][col];
		}
	}
	return grid;
}



