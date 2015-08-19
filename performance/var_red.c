#include <stdio.h>

void myprint(void);

float var_red(int N, float* arr);


void myprint()
{
    printf("hello world\n");
}


float var_red(int N, float* arr)
{
	float sum = 0.0;
	float Nf = N;

	for (int i = 0; i < N; i++) {
		for (int j = i + 1; j < N; j++) {
			sum += (arr[i] - arr[j]) * (arr[i] - arr[j]);
		}
	}

	float res = sum / (Nf * Nf) ;
	// printf("Get N=%d %f -> res=%f\n", N, arr[0], res);
	return res;
}
