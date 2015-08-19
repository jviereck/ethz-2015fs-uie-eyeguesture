#include <stdio.h>

#include <x86intrin.h>

void myprint(void);

float var_red(int N, float* arr);


void myprint()
{
    printf("hello world\n");
}


float var_red(int N, float* arr)
{
  // RECALL: 128 -> 4 float - 256 -> 8 float
  __m256 index, x_vec;



  float sum = 0.0;
  float Nf = N;

  int N8 = (N / 8) * 8;

  for (int i = 0; i < N8; i++) {
    // void _mm256_storeu_ps (float * mem_addr, __m256 a)
    for (int j = i + 1; j < N; j++) {


      sum += (arr[i] - arr[j]) * (arr[i] - arr[j]);

      // __m256 _mm256_set1_ps (float a)
      // _mm256_storeu_ps(float *, __m256)
      // _mm256_add_ps(__m256, __m256)
      // __m256 _mm256_mul_ps (__m256 a, __m256 b)  // Multiplication
    }
  }

  for (int i = N8; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      sum += (arr[i] - arr[j]) * (arr[i] - arr[j]);
    }
  }

  float res = sum / (Nf * Nf) ;
  // printf("Get N=%d %f -> res=%f\n", N, arr[0], res);
  return res;
}
