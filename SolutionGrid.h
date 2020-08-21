#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_BLOCK 32
#define NUM_THREAD 64
#define NUM_EMPTY 27

using namespace std;

class SolutionGrid{
	int* soln_grid;
	SolutionGrid() {
	}
};

int* sudokuInit(int* grid, int** config, int grid_size);

__global__ void checkValid(int* grid, short* veri_row, short* veri_col, short* veri_box, int* grid_size, int* box_size);
__global__ void genPoss(int* grid, short* poss, short* veri_row, short* veri_col, short* veri_box, int* soln_row, int* soln_col, int* soln_val, int* grid_size, int* box_size, int* soln_size);
__global__ void solveCuda(int* grid, int* soln_row, int* soln_col, int* soln_val, short* poss, int* grid_size, int* soln_size, int* box_size);

__device__ void checkRow(int* grid, short* veri_row, int grid_size, int row_s, int row_e);
__device__ void checkCol(int* grid, short* veri_col, int grid_size, int col_s, int col_e);
__device__ void checkBox(int* grid, short* veri_box, int grid_size, int box_size, int thd_cnt_s, int thd_cnt_e);

__device__ int calcThreadWidth(int grid_size);
__device__ int calcStartIdx(int grid_size);
__device__ int calcBlockWidth(int grid_size);
__device__ int calcStartIdxBlock(int grid_size);

__device__ void swap(int* soln, int idx1, int idx2);
__device__ void swap(short* soln, int idx1, int idx2);
__device__ int getBox(int row, int col, int box_size);

__device__ void setRelativeVal(short* l_poss, int val, int curr_idx, int soln_cnt, int* soln_row, int* soln_col, int grid_size, int box_size, int soln_size);
__device__ void resetRelativeVal(short* l_poss_curr, short* l_poss, int curr_idx, int soln_cnt, int* soln_row, int* soln_col, int grid_size, int box_size, int soln_size);
