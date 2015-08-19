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

  __m256 vsum;
  vsum = _mm256_set1_ps(0.0);

  for (int i = 0; i < N; i++) {
    float arri = arr[i];
    __m256 ival = _mm256_set1_ps(arri);

    int j = i + 1;
    for (j; j < N - 7; j += 8) {
      // Work always on the next 8 blocks at the same time.

      // for (int k = 0; k < 8; k++) {
      //   int q = j + k;
      //   sum += (arr[i] - arr[q]) * (arr[i] - arr[q]);
      // }

      // These SIMD instructions are doing 1:1 the same as the above lines -
      // however, instead of looping over the elements, the computation is done
      // in one pass using 8-way vector operations ;)
      __m256 entry = _mm256_loadu_ps(arr + j);
      __m256 diff = _mm256_sub_ps(ival, entry);
      __m256 vsquare = _mm256_mul_ps(diff, diff);
      vsum = _mm256_add_ps(vsum, vsquare);
    }

    for (j; j < N; j++) {
      float t = (arri - arr[j]);
      sum += t * t;
    }
  }

  // Sum up the accumulated values from 'vsum' vector.
  // There are 'smarter' ways in doing this (that rely in vector instructions
  // again), but it turns out this version is much slower than the one
  // proposed in https://software.intel.com/en-us/forums/topic/346695 and as this
  // code runs only once per function invocation, the overhead is not significant.
  for (int i = 0; i < 8; i++) {
    sum += vsum[i];
  }

  return sum / Nf;
}
