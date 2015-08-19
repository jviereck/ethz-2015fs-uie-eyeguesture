#include <stdio.h>

#include <x86intrin.h>

void myprint(void);

float var_red(int N, float* arr);


void myprint()
{
    printf("hello world\n");
}

// As based on https://software.intel.com/en-us/forums/topic/346695
void HsumAvxFlt(const float * const adr) {

  float sumAVX = 0;

  __m256 avx = _mm256_loadu_ps(&adr[0]);
  __m256 hsum = _mm256_hadd_ps(avx, avx);

  hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));

  _mm_store_ss(&sumAVX, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
  printf("Hsum AVX Double: %f\n", sumAVX);
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
    int j = i + 1;
    for (j; j < N - 7; j += 8) {
      // Work always on the next 8 blocks at the same time.
      for (int k = 0; k < 8; k++) {
        int q = j + k;
        sum += (arr[i] - arr[q]) * (arr[i] - arr[q]);
      }
    }

    for (j; j < N; j++) {
      sum += (arr[i] - arr[j]) * (arr[i] - arr[j]);
    }
  }

  // for (int i = 0; i < N8; i++) {
  //   __m256 ival;

  //   _mm256_storeu_ps(arr + i, ival);


  //   // void _mm256_storeu_ps (float * mem_addr, __m256 a)
  //   for (int j = i ; j < N; j++) {
  //     __m256 entry = _mm256_set1_ps(arr[j])

  //     vsquare =
  //     vsum = _mm256_add_ps(vsum, vsquare)

  //     sum += (arr[i] - arr[j]) * (arr[i] - arr[j]);

  //     // __m256 _mm256_set1_ps (float a)
  //     // _mm256_storeu_ps(float *, __m256)
  //     // _mm256_add_ps(__m256, __m256)
  //     // __m256 _mm256_mul_ps (__m256 a, __m256 b)  // Multiplication
  //   }

  // }

  // for (int i = N8; i < N; i++) {
  //   for (int j = i + 1; j < N; j++) {
  //     sum += (arr[i] - arr[j]) * (arr[i] - arr[j]);
  //   }
  // }

  float res = sum / (Nf * Nf) ;
  // printf("Get N=%d %f -> res=%f\n", N, arr[0], res);
  return res;
}
