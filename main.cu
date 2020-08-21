#include "SolutionGrid.h"
#include "SudokuGrid.h"

__global__ void genPoss(int* grid, short* poss, short* veri_row, short* veri_col, short* veri_box, int* soln_row, int* soln_col, int* soln_val, int* grid_size, int* box_size, int* soln_size) {
	if (blockIdx.x < *soln_size) {
		int b_width = calcBlockWidth(*soln_size);
		int b_s_idx = calcStartIdxBlock(*soln_size);
		int b_e_idx = b_s_idx + b_width;
		if (threadIdx.x < *grid_size) {
			//printf("blk %d bs %d be %d bw %d\n", blockIdx.x, b_s_idx, b_e_idx, b_width);
			int t_width = calcThreadWidth(*grid_size);
			int s_idx = calcStartIdx(*grid_size);
			int e_idx = s_idx + t_width;
			for (int blk_cnt = b_s_idx; blk_cnt < b_e_idx; blk_cnt++) { // row idx of poss
				for (int thd_cnt = s_idx; thd_cnt < e_idx; thd_cnt++) { // column idx of poss
					int row = soln_row[blk_cnt];
					int col = soln_col[blk_cnt];
					int box = getBox(row, col, *box_size);
					int veri_row_idx = row * (*grid_size) + thd_cnt;
					int veri_col_idx = col * (*grid_size) + thd_cnt;
					int veri_box_idx = box * (*grid_size) + thd_cnt;
					poss[blk_cnt * (*grid_size) + thd_cnt] = veri_row[veri_row_idx] + veri_col[veri_col_idx] + veri_box[veri_box_idx];
				}
			}
			__syncthreads();
		}
	}
}

__global__ void checkValid(int* grid, short* veri_row, short* veri_col, short* veri_box, int* grid_size, int* box_size) {
	if (threadIdx.x < *grid_size) {
		int t_width = calcThreadWidth(*grid_size);
		int s_idx = calcStartIdx(*grid_size);
		int e_idx = s_idx + t_width;
		
		checkRow(grid, veri_row, *grid_size, s_idx, e_idx);
		checkCol(grid, veri_col, *grid_size, s_idx, e_idx);
		checkBox(grid, veri_box, *grid_size, *box_size, s_idx, e_idx);
		__syncthreads();
	}
}

__global__ void solveCuda(int* grid, int* soln_row, int* soln_col, int* soln_val, short* poss, int* grid_size, int* soln_size, int* box_size) {
	__shared__ short l_poss[NUM_EMPTY * 9];
	__shared__ short l_poss_curr[NUM_EMPTY * 9];
	__shared__ short max_poss[NUM_EMPTY];
	__shared__ int l_soln_val[NUM_EMPTY];
	__shared__ short max;
	__shared__ int max_idx;
	if (blockIdx.x < *grid_size) {
		//printf("blk %d bs %d be %d bw %d\n", blockIdx.x, b_s_idx, b_e_idx, b_width);
		//printf("blk %d, thd %d\n", blockIdx.x, threadIdx.x);
		if (threadIdx.x < (*grid_size)) {
			int t_width = calcThreadWidth(*grid_size);
			int s_idx = calcStartIdx(*grid_size);
			int e_idx = s_idx + t_width;
			//printf("blk %d, thd %d\n", blockIdx.x, threadIdx.x);
			for (int sol_cnt = 0; sol_cnt < (*soln_size); sol_cnt++) {
				max_poss[sol_cnt] = 0;
				l_soln_val[sol_cnt] = 0;
				for (int thd_cnt = s_idx; thd_cnt < e_idx; thd_cnt++) {
					int poss_idx = sol_cnt * (*grid_size) + thd_cnt;
					l_poss[poss_idx] = poss[poss_idx];
					l_poss_curr[poss_idx] = l_poss[poss_idx];
//					if (blockIdx.x == 0) {
//						printf("[%d, %d] %d l_poss %d poss %d\n", sol_cnt, thd_cnt, poss_idx, l_poss[poss_idx], poss[poss_idx]);
//					}
				}
			}
			__syncthreads();
		}


		if (threadIdx.x < (*soln_size)) {
			int t_width_sol = calcThreadWidth(*soln_size);
			int s_idx_sol = calcStartIdx(*soln_size);
			int e_idx_sol = s_idx_sol + t_width_sol;
//			if (blockIdx.x == 0)
//				printf("s %d e %d w %d\n", s_idx_sol, e_idx_sol, t_width_sol);
			for (int sol_cnt = s_idx_sol; sol_cnt < e_idx_sol; sol_cnt++) {
				for (int thd_cnt = 0; thd_cnt < (*grid_size); thd_cnt++) {
					int poss_idx = sol_cnt * (*grid_size) + thd_cnt;
//					if (blockIdx.x == 0)
//						printf("[%d, %d] %d max_poss[%d] %d\n", sol_cnt, thd_cnt, poss_idx, sol_cnt, max_poss[sol_cnt]);
					if (l_poss[poss_idx] == 0) {
						//printf("max_poss %d\n", max_poss[sol_cnt]);
						max_poss[sol_cnt] = max_poss[sol_cnt] + 1;
					}
				}
			}

			__syncthreads();
		}

//			if (blockIdx.x == 0 && threadIdx.x == 0) {
//				for (int i = 0; i < (*soln_size); i++) {
//					printf("[%d] \t[%d, %d] ", i, soln_row[i], soln_col[i]);
//					for (int j = 0; j < (*grid_size); j++) {
//						printf("[%d] %d ", j, l_poss[i * (*grid_size) + j]);
//					}
//					printf("\n");
//				}
//				printf("\n");
//				for (int idx = 0; idx < (*soln_size); idx++)
//					printf("[%d] %d ,", idx, max_poss[idx]);
//				printf("\n");
//			}


		if (threadIdx.x == 0) {
			max = 0;
			for (int idx = 0; idx < (*soln_size); idx++) {
				if (max_poss[idx] > max) {
					max = max_poss[idx];
					max_idx = idx;
				}
			}
//			printf("max poss [%d] %d\n", max_idx, max);
		}

		if (blockIdx.x == 0 && threadIdx.x == 0) {
			swap(soln_row, 0, max_idx);
			swap(soln_col, 0, max_idx);
		}


		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//	for (int i = 0; i < (*soln_size); i++) {
		//		printf("[%d] \t[%d, %d] ", i, soln_row[i], soln_col[i]);
		//		for (int j = 0; j < (*grid_size); j++) {
		//			printf("[%d] %d ", j, l_poss[i * (*grid_size) + j]);
		//		}
		//		printf("\n");
		//	}
		//	printf("\n");
		//	for (int idx = 0; idx < (*soln_size); idx++)
		//		printf("[%d] %d ,", idx, max_poss[idx]);
		//	printf("\n");
		//}

		if (threadIdx.x < (*grid_size)) {
			int t_width = calcThreadWidth(*grid_size);
			int s_idx = calcStartIdx(*grid_size);
			int e_idx = s_idx + t_width;
			for (int thd_cnt = s_idx; thd_cnt < e_idx; thd_cnt++) {
				int poss_idx_1 = thd_cnt;
				int poss_idx_2 = max_idx * (*grid_size) + thd_cnt;
				swap(l_poss, poss_idx_1, poss_idx_2);
				swap(l_poss_curr, poss_idx_1, poss_idx_2);
			}
			__syncthreads();
		}

		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//	for (int j = 0; j < (*soln_size); j++) {
		//		printf("curr %d [%d, %d] [%d] ", j, soln_row[j], soln_col[j], getBox(soln_row[j], soln_col[j], *box_size));
		//		for (int i = 0; i < (*grid_size); i++) {
		//			printf("[%d]%d", i, l_poss_curr[j * (*grid_size) + i]);
		//		}
		//		printf("\n");
		//	}
		//}

		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//	for (int i = 0; i < (*soln_size); i++) {
		//		printf("[%d] [%d, %d] %d\n", i, soln_row[i], soln_col[i], max_poss[i]);
		//	}
		//}

		__syncthreads();
	}

	if (blockIdx.x < max) {
		int b_width = calcBlockWidth(max);
		int b_s_idx = calcStartIdxBlock(max);
		int b_e_idx = b_s_idx + b_width;

		if (threadIdx.x == 0) {
			for (int blk_cnt = b_s_idx; blk_cnt < b_e_idx; blk_cnt++) {
				bool branching = true;
				for (int curr_idx = 0; curr_idx < (*soln_size); curr_idx++) {
					bool soln_found = false;
					int poss_cnt = -1;
					//if (blockIdx.x == 0) {
					//	printf("\ncurr-1 %d [%d, %d]: ", curr_idx - 1, soln_row[curr_idx - 1], soln_col[curr_idx - 1]);
					//	for (int i = 0; i < (*grid_size); i++) {
					//		printf("[%d]%d", i, l_poss_curr[(curr_idx - 1) * (*grid_size) + i]);
					//	}
					//	printf("\n");

					//	printf("curr   %d [%d, %d]: ", curr_idx, soln_row[curr_idx], soln_col[curr_idx]);
					//	for (int i = 0; i < (*grid_size); i++) {
					//		printf("[%d]%d", i, l_poss_curr[curr_idx * (*grid_size) + i]);
					//	}
					//	printf("\n");

					//	for (int i = 0; i < (*soln_size); i++) {
					//		printf("[%d]%d ", i, l_soln_val[i]);
					//	}
					//	printf("\n");

					//}

					for (int soln_cnt = 0; soln_cnt < (*grid_size); soln_cnt++) {
						int poss_idx = curr_idx * (*grid_size) + soln_cnt;
						if (l_poss_curr[poss_idx] == 0) {
							poss_cnt++;
							if (branching) {
								branching = blk_cnt != poss_cnt;
							}
							if (!branching) {
								// check existing solutions
								bool exist = false;
								for (int chk_soln = 0; l_soln_val[chk_soln] != 0 && chk_soln < (*soln_size); chk_soln++) {
									if (soln_row[chk_soln] == soln_row[curr_idx]) {
										if (l_soln_val[chk_soln] == soln_cnt + 1) {
											exist = true;
											break;
										}
									}
									if (soln_col[chk_soln] == soln_col[curr_idx]) {
										if (l_soln_val[chk_soln] == soln_cnt + 1) {
											exist = true;
											break;
										}
									}
									int soln_box = getBox(soln_row[curr_idx], soln_col[curr_idx], *box_size);
									int chk_box = getBox(soln_row[chk_soln], soln_col[chk_soln], *box_size);
									if (soln_box == chk_box) {
										if (l_soln_val[chk_soln] == soln_cnt + 1) {
											exist = true;
											break;
										}
									}
								}
								if (!exist) {
									soln_found = true;
									l_poss_curr[poss_idx] = 1;
									l_soln_val[curr_idx] = soln_cnt + 1;
									break;
								}
							}
						}
					}

					if (!soln_found) {
						if (curr_idx < 1) {
							break;
						}

						for (int reset_cnt = 0; reset_cnt < (*grid_size); reset_cnt++) {
							int reset_idx = curr_idx * (*grid_size) + reset_cnt;
							l_poss_curr[reset_idx] = l_poss[reset_idx];
						}
						l_soln_val[curr_idx - 1] = 0;
						curr_idx -= 2;
					}
					if (curr_idx < 1 && !branching) {
						break;
					}
					branching = false;

				}
				//for (int i = 0; i < *soln_size; i++) {
				//	printf("blk %d [%d](%d %d) %d\n", blk_cnt, i, soln_row[i], soln_col[i], l_soln_val[i]);
				//}
			}
		}

//			if (threadIdx.x == 0) {
//				branching = true;
//				for (curr_idx = 0; curr_idx < (*soln_size); curr_idx++) { // iterate over all solutions
//					if (blockIdx.x == 0) {
//						printf("\ncurr-1 %d [%d, %d]: ", curr_idx - 1, soln_row[curr_idx - 1], soln_col[curr_idx - 1]);
//						for (int i = 0; i < (*grid_size); i++) {
//							printf("[%d]%d", i, l_poss_curr[(curr_idx - 1) * (*grid_size) + i]);
//						}
//						printf("\n");
//
//						printf("curr   %d [%d, %d]: ", curr_idx, soln_row[curr_idx], soln_col[curr_idx]);
//						for (int i = 0; i < (*grid_size); i++) {
//							printf("[%d]%d", i, l_poss_curr[curr_idx * (*grid_size) + i]);
//						}
//						printf("\n");
//					}
//					soln_found = false;
//					int poss_cnt = -1;
//					for (int soln_cnt = 0; soln_cnt < (*grid_size); soln_cnt++) { // the jth number of a cell
//						int poss_idx = curr_idx * (*grid_size) + soln_cnt;
//						if (l_poss_curr[poss_idx] == 0) {
//							poss_cnt++;
//							if (!branching || blk_cnt == poss_cnt) {
//								soln_found = true;
//								setRelativeVal(l_poss_curr, 1, curr_idx, soln_cnt, soln_row, soln_col, *grid_size, *box_size, *soln_size);
//								l_poss_curr[poss_idx] = 1;
//								l_soln_val[curr_idx] = soln_cnt + 1;
//								break;
//							}
//						}
//					}
//
//					// backward case
//					if (!soln_found) {
//						stop_at = curr_idx;
//
//						if (curr_idx < 1) // have a branch checked exhaustively
//							break;
//						for (int reset_cnt = 0; reset_cnt < (*grid_size); reset_cnt++) {
//							int reset_idx = curr_idx * (*grid_size) + reset_cnt;
//							l_poss_curr[reset_idx] = poss[reset_idx];
//							//printf("resetting %d\n", reset_cnt);
//							resetRelativeVal(l_poss_curr, poss, curr_idx, reset_cnt, soln_row, soln_col, *grid_size, *box_size, *soln_size);
//						}
//						if (blockIdx.x == 0 && threadIdx.x == 0) {
//							printf("reset:");
//							printf("curr   %d [%d, %d]: ", curr_idx, soln_row[curr_idx], soln_col[curr_idx]);
//							for (int i = 0; i < (*grid_size); i++) {
//								printf("[%d]%d", i, l_poss_curr[curr_idx * (*grid_size) + i]);
//							}
//							printf("\n");
//						}
//						curr_idx --; // go back one cell and clear possiblity
//
//						//for (int reset_cnt = 0; reset_cnt < (*grid_size); reset_cnt++) {
//						//	int reset_idx = curr_idx * (*grid_size) + reset_cnt;
//						//	l_poss[reset_idx] = l_poss_curr[reset_idx];
//						//	resetRelativeVal(l_poss_curr, l_poss, curr_idx, reset_cnt, soln_row, soln_col, *grid_size, *box_size, *soln_size);
//						//}
//						//l_soln_val[curr_idx] = 0;
//						curr_idx --; // nullify increment of for loop
//					}
//
//					if (curr_idx < 1 && !branching) {
//						break;
//					}
//					branching = false;
//				}
//			}
//		}
//		__syncthreads();
		if (l_soln_val[(*soln_size) - 1] != 0) {
			if (threadIdx.x < *soln_size) {
				int t_width = calcThreadWidth(*soln_size);
				int s_idx = calcStartIdx(*soln_size);
				int e_idx = s_idx + t_width;
				for (int thd_cnt = s_idx; thd_cnt < e_idx; thd_cnt++) {
					soln_val[thd_cnt] = l_soln_val[thd_cnt];
				}
				for (int thd_cnt = s_idx; thd_cnt < e_idx; thd_cnt++) {
					int g_idx = soln_row[thd_cnt] * (*grid_size) + soln_col[thd_cnt];
					grid[g_idx] = l_soln_val[thd_cnt];
					//printf("%d %d %d %d\n", soln_row[thd_cnt], soln_col[thd_cnt], g_idx, grid[g_idx]);
				}
				__syncthreads();
				if (threadIdx.x == 0) {
					for (int i = 0; i < *grid_size; i++) {
						for (int j = 0; j < *grid_size; j++) {
							int g_idx = i * (*grid_size) + j;
							printf("%d ", grid[g_idx]);
						}
						printf("\n");
					}
				}
			}
		}
		__syncthreads();

		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//	for (int i = 0; i < *soln_size; i++) {
		//		printf("[%d] %d\n", i, soln_val[i]);
		//	}
		//	for (int i = 0; i < *grid_size; i++) {
		//		for (int j = 0; j < *grid_size; j++) {
		//			int g_idx = i * (*grid_size) + j;
		//			printf("%d ", grid[g_idx]);
		//		}
		//		printf("\n");
		//	}
		//}
	}
}

__device__ void resetRelativeVal(short* l_poss_curr, short * l_poss, int curr_idx, int soln_cnt, int* soln_row, int* soln_col, int grid_size, int box_size, int soln_size) {
	int curr_row = soln_row[curr_idx];
	int curr_col = soln_col[curr_idx];
	int curr_box = getBox(curr_row, curr_col, box_size);
	for (int idx = curr_idx; idx < soln_size; idx++) {
		if (idx != curr_idx) {
			int chg_idx = idx * grid_size + soln_cnt;
			if (soln_row[idx] == curr_row) {
				l_poss_curr[chg_idx] = l_poss[chg_idx];
			}
			if (soln_col[idx] == curr_col) {
				l_poss_curr[chg_idx] = l_poss[chg_idx];
			}
			if (getBox(soln_row[idx], soln_col[idx], box_size) == curr_box) {
				l_poss_curr[chg_idx] = l_poss[chg_idx];
			}
		}
	}
}

__device__ void setRelativeVal(short* l_poss, int val, int curr_idx, int soln_cnt, int* soln_row, int* soln_col, int grid_size, int box_size, int soln_size) {
	int curr_row = soln_row[curr_idx];
	int curr_col = soln_col[curr_idx];
	int curr_box = getBox(curr_row, curr_col, box_size);
	for (int idx = curr_idx; idx < soln_size; idx++) {
		int chg_idx = idx * grid_size + soln_cnt;
		int soln_box = getBox(soln_row[idx], soln_col[idx], box_size);
		if (soln_row[idx] == curr_row) {
			l_poss[chg_idx] = val;
		}
		if (soln_col[idx] == curr_col) {
			l_poss[chg_idx] = val;
		}
		if (soln_box == curr_box) {
			l_poss[chg_idx] = val;
		}
	}
}

__device__ int getBox(int row, int col, int box_size) {
	int box = row / box_size * box_size + col / box_size;
	return box;
}

__device__ void swap(int* soln, int idx1, int idx2) {
	int temp = soln[idx1];
	soln[idx1] = soln[idx2];
	soln[idx2] = temp;
}

__device__ void swap(short* soln, int idx1, int idx2) {
	short temp = soln[idx1];
	soln[idx1] = soln[idx2];
	soln[idx2] = temp;
}

__device__ void checkRow(int* grid, short* veri_row, int grid_size, int row_s, int row_e) {
	for (int row = row_s; row < row_e; row++) {
		for (int col = 0; col < grid_size; col++) {
			int g_idx = row * grid_size + col;
			int soln_idx = row * grid_size + grid[g_idx];
			if (grid[g_idx] != 0)
				veri_row[soln_idx - 1] = 1;
		}
	}
}

__device__ void checkCol(int* grid, short* veri_col, int grid_size, int col_s, int col_e) {
	for (int row = 0; row < grid_size; row++) {
		for (int col = col_s; col < col_e; col++) {
			int g_idx = row * grid_size + col;
			int soln_idx = col * grid_size + grid[g_idx];
			if (grid[g_idx] != 0)
				veri_col[soln_idx - 1] = 1;
		}
	}
}

__device__ void checkBox(int* grid, short* veri_box, int grid_size, int box_size, int thd_cnt_s, int thd_cnt_e) {
	for (int thd_cnt = thd_cnt_s; thd_cnt < thd_cnt_e; thd_cnt++) {
		int box_row_s = thd_cnt * box_size / grid_size * box_size; // row start location of box
		int box_row_e = box_row_s + box_size;
		int box_col_s = thd_cnt * box_size - thd_cnt * box_size / grid_size * grid_size;
		int box_col_e = box_col_s + box_size;
//		printf("thd %d r_s %d r_e %d c_s %d c_e %d\n", threadIdx.x, box_row_s, box_row_e, box_col_s, box_col_e);
		for (int row = box_row_s; row < box_row_e; row++) {
			for (int col = box_col_s; col < box_col_e; col++) {
				int g_idx = row * grid_size + col;
//				if(threadIdx.x == 0)
//					printf("thread %d [%d][%d] %d \n",threadIdx.x, row, col, g_idx);
				int soln_idx = thd_cnt * grid_size + grid[g_idx];
				if (grid[g_idx] != 0)
					veri_box[soln_idx - 1] = 1;
			}
		}
	}
}

__device__ int calcBlockWidth(int grid_size) {
	int b_width;
	if (gridDim.x < grid_size) {
		b_width = (grid_size) / gridDim.x;
		if (blockIdx.x > gridDim.x - 2)
			b_width = grid_size - blockIdx.x * b_width;
	}
	else {
		b_width = 1;
	}
	return b_width;
}

__device__ int calcStartIdxBlock(int grid_size) {
	int s_idx;
	if (gridDim.x < grid_size) {
		s_idx = blockIdx.x * (grid_size / gridDim.x);
	}
	else {
		s_idx = blockIdx.x;
	}
	return s_idx;
}

__device__ int calcThreadWidth(int grid_size) {
	int t_width;
	if (blockDim.x < grid_size) {
		t_width = (grid_size) / blockDim.x;
		if (threadIdx.x > blockDim.x - 2)
			t_width = grid_size - threadIdx.x * t_width;
	}
	else {
		t_width = 1;
	}
	return t_width;
}

__device__ int calcStartIdx(int grid_size) {
	int s_idx;
	if (blockDim.x < grid_size) {
		s_idx = threadIdx.x * (grid_size / blockDim.x);
	}
	else {
		s_idx = threadIdx.x;
	}
	return s_idx;
}

int main(int argc, char ** argv) {
	int grid_size = stoi(argv[1]);
	//int zeros = grid_size * grid_size / 3;
	int zeros = NUM_EMPTY;
	int* soln_val = new int[zeros];
	int* soln_row = new int[zeros];
	int* soln_col = new int[zeros];

	SudokuGrid config = SudokuGrid(grid_size);
	config.randFirstRow(&config.grid, grid_size);
	config.solve(&config.grid, grid_size);
	//config.printGrid();
	config.randClearSlots(&config.grid, grid_size, zeros, soln_row, soln_col, soln_val);
	config.printGrid();

	cout << "Number of empty slots = " << zeros << endl;

	// implement in serial by CPU

	time_t start_time = time(NULL);
	//config.solve(&config.grid, grid_size);
	time_t end_time = time(NULL);
	printf("Time spent: %ld\n", end_time - start_time);
	//config.randClearSlots(&config.grid, grid_size, zeros, soln_row, soln_col, soln_val);

	// CUDA parts starts
	cudaError_t cuda_status;
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;

	int grid_size_cuda = grid_size * grid_size;
	int box_size = sqrt(grid_size);

	int* grid = sudokuInit(NULL, config.grid, grid_size);


	int* _grid = 0;
	int* _soln_val = 0;
	int* _soln_row = 0;
	int* _soln_col = 0;

	short* _veri_row = 0;
	short* _veri_col = 0;
	short* _veri_box = 0;
	short* _poss = 0;

	int* _grid_size = 0;
	int* _box_size = 0;
	int* _soln_size = 0;

	
	cuda_status = cudaMalloc((void**)&_grid, grid_size_cuda * sizeof(int));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc grid failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_soln_val, zeros * sizeof(int));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc soln_val failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_soln_row, zeros * sizeof(int));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc soln_row failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_soln_col, zeros * sizeof(int));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc soln_col failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_veri_row, grid_size_cuda * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc veri_row failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_veri_col, grid_size_cuda * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc veri_col failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_veri_box, grid_size_cuda * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc veri_box failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_poss, zeros * grid_size * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc poss failed with error code " << cuda_status << endl;
	}

	cuda_status = cudaMalloc((void**)&_grid_size, sizeof(int));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc grid_size failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_box_size, sizeof(int));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc box_size failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMalloc((void**)&_soln_size, sizeof(int));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMalloc soln_size failed with error code " << cuda_status << endl;
	}
	
	cuda_status = cudaMemset(_veri_row, 0, grid_size_cuda * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemset veri_row failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemset(_veri_col, 0, grid_size_cuda * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemset veri_col failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemset(_veri_box, 0, grid_size_cuda * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemset veri_box failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemset(_poss, 0, zeros * grid_size * sizeof(short));
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemset poss failed with error code " << cuda_status << endl;
	}
	
	cuda_status = cudaMemcpy(_grid, grid, grid_size_cuda * sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemcpy grid failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemcpy(_soln_val, soln_val, zeros * sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemcpy soln_val failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemcpy(_soln_row, soln_row, zeros * sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemcpy soln_row failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemcpy(_soln_col, soln_col, zeros * sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemcpy soln_col failed with error code " << cuda_status << endl;
	}

	cuda_status = cudaMemcpy(_grid_size, &grid_size, sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemcpy grid_size failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemcpy(_box_size, &box_size, sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemcpy box_size failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaMemcpy(_soln_size, &zeros, sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		cerr << "cudaMemcpy soln_size failed with error code " << cuda_status << endl;
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//solveCuda <<<1, NUM_THREAD >>> (_grid, _soln_row, _soln_col, _soln_val, _veri_row, _veri_col, _veri_box, _grid_size, _box_size);
	checkValid <<<1, NUM_THREAD>>> (_grid, _veri_row, _veri_col, _veri_box, _grid_size, _box_size);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		cerr << "checkValid launch failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaDeviceSynchronize();

	genPoss <<<NUM_BLOCK, NUM_THREAD >>> (_grid, _poss, _veri_row, _veri_col, _veri_box, _soln_row, 
		_soln_col, _soln_val, _grid_size, _box_size, _soln_size);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		cerr << "genPoss launch failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaDeviceSynchronize();


	
//	solveCudaHost(grid, soln_row, soln_col, soln_val, poss, poss_curr, grid_size, zeros, box_size);
//	for (int i = 0; i < zeros; i++) {
//		printf("[%d](%d, %d) %d\n", i, soln_row[i], soln_col[i], soln_val[i]);
//	}

	solveCuda <<<1, NUM_THREAD >>> (_grid, _soln_row, _soln_col, _soln_val, _poss, _grid_size, _soln_size, _box_size);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		cerr << "solveCuda launch failed with error code " << cuda_status << endl;
	}
	cuda_status = cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("Time spent: %.5f\n", gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	cudaFree(_grid);
	cudaFree(_soln_val);
	cudaFree(_soln_row);
	cudaFree(_soln_col);
	cudaFree(_veri_row);
	cudaFree(_veri_col);
	cudaFree(_veri_box);

	delete[] grid;
	delete[] soln_val;
	delete[] soln_col;
	delete[] soln_row;
	return EXIT_SUCCESS;
}
