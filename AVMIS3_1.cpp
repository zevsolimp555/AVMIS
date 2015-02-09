#include<stdio.h>
#include<stdlib.h>
#include<time.h>

/*Intrinsics*/
#include<mmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>

#define MATRIX_SIZE 1200

void matrixMulVectorization();
void matrixMulNoVectorization();
void matrixMulIntrinsics();

int main()
{
	/*No vectorization*/
	time_t runTime = clock();
	time_t runTime2;
	matrixMulNoVectorization();
	runTime2 = clock() - runTime;
	printf("1. %.3f s.", (float)runTime2/CLOCKS_PER_SEC);

	/*Vectorization*/
	runTime = clock();
	matrixMulVectorization();
	runTime2 = clock() - runTime;
	printf("\n2. %.3f s.", (float)runTime2/CLOCKS_PER_SEC);

	/*Intrinsics*/
	runTime = clock();
	matrixMulIntrinsics();
	runTime2 = clock() - runTime;
	printf("\n3. %.3f s.", (float)runTime2/CLOCKS_PER_SEC);

	return 0;
}

void matrixMulVectorization()
{
	float * matrix1 = (float *)_aligned_malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 16);
	float * matrix2 = (float *)_aligned_malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 16);
	float * matrix3 = (float *)_aligned_malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 16);

	float * matrix3temp;
	float * matrix2temp;

	for(int i = 0; i < MATRIX_SIZE; i++)
	{
		for(int k = 0; k < MATRIX_SIZE; k++)
		{
			float temp = matrix1[i * MATRIX_SIZE + k];
			matrix3temp = matrix3 + i * MATRIX_SIZE;
			matrix2temp = matrix2 + k * MATRIX_SIZE;
			for(int j = 0; j < MATRIX_SIZE; j++)
				matrix3temp[j] += temp * matrix2temp[j];
		}
	}

	_aligned_free(matrix1);
	_aligned_free(matrix2);
	_aligned_free(matrix3);
}

void matrixMulNoVectorization()
{
	float * matrix1 = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
	float * matrix2 = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
	float * matrix3 = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

	float * matrix3temp;
	float * matrix2temp;

	for(int i = 0; i < MATRIX_SIZE; i++)
	{
		for(int k = 0; k < MATRIX_SIZE; k++)
		{
			float temp = matrix1[i * MATRIX_SIZE + k];
			matrix3temp = matrix3 + i * MATRIX_SIZE;
			matrix2temp = matrix2 + k * MATRIX_SIZE;
#pragma loop(no_vector)
			for(int j = 0; j < MATRIX_SIZE; j++)
				matrix3temp[j] += temp * matrix2temp[j];
		}
	}

	free(matrix1);
	free(matrix2);
	free(matrix3);
}

void matrixMulIntrinsics()
{
	float * matrix1 = (float *)_aligned_malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 16);
	float * matrix2 = (float *)_aligned_malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 16);
	float * matrix3 = (float *)_aligned_malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 16);

	__m128 * matrix3temp;
	__m128 * matrix2temp;
	__m128 mulTemp;

	for(int i = 0; i < MATRIX_SIZE; i++)
	{
		for(int k = 0; k < MATRIX_SIZE; k++)
		{
			__m128 temp = _mm_set_ps1(matrix1[i * MATRIX_SIZE + k]);
			matrix3temp = (__m128 *)(matrix3 + i * MATRIX_SIZE);
			matrix2temp = (__m128 *)(matrix2 + k * MATRIX_SIZE);

			for(int j = 0; j < MATRIX_SIZE / 4; j++)
			{
				mulTemp = _mm_mul_ps(temp, *(matrix2temp + j));
				*(matrix3temp + j) = _mm_add_ps(mulTemp, *(matrix3temp + j));
			}
		}
	}

	_aligned_free(matrix1);
	_aligned_free(matrix2);
	_aligned_free(matrix3);
}