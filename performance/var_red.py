import numpy as np

# NOTE: See the StackOverflow example on how to get started with ctypes
#    in python! -> http://stackoverflow.com/a/5082294
import ctypes
var_red_lib = ctypes.CDLL('var_red.so')

# IMPORTANT: Must specify the return type of the c-function. Otherwise python
#    doesn't really pick it up correctly and the result is garbage!
var_red_lib_fn = var_red_lib.var_red
var_red_lib_fn.restype = ctypes.c_float


def run_test(var_red_fn):
  # Small helper function to generate an np.array with the right type from the
  # plain python arrays.
  wrap = lambda a: np.array(a, dtype='f32');

  a = wrap([-10.4638427, -20.4298427, -13.7448427,  17.5441573,   1.0081573,  -4.8018427
    -6.5888427,  33.6321573, -13.2928427, -14.6338427])


  b = wrap([   3.07846067,  12.46446067,  -7.64653933,   7.10146067,   6.65446067,
    -4.07153933,  -3.62453933,  -4.51853933, -21.94753933,   3.97246067,
    -3.17753933,  41.06646067,  39.27846067,  41.51346067, -87.64353933,
   -22.84153933, -27.31053933, -35.35553933, -47.42153933, -53.23153933,
   -41.16453933, -30.43953933,   7.10146067,  32.12746067,  34.80946067,
   -18.37253933, -21.94753933,  -9.88153933,  -5.85953933,   3.97246067,
     1.73846067,  -7.64653933, -11.66953933, -11.22253933, -19.26653933,
    28.10546067,  58.94246067,  40.61946067,  21.40246067,  66.98646067,
    48.21646067,  22.74246067,  -3.17753933,  -0.94353933,  -6.75353933,
    16.93346067])

  c = wrap([ 13.68134831,  19.37634831,  43.08434831,  66.62934831, -24.09265169,
   -14.86297753, -11.73497753,  -7.26597753,   4.35402247,   9.27002247,
     1.22602247,  -3.24297753,  -6.81897753, -10.84097753, -65.39097753,
    -6.21665169, -19.62365169, -23.64565169, -39.73465169, -40.18165169])

  d = wrap([-42.12397753,  57.08902247,  44.57602247,  36.08402247,  16.42002247,
    55.30102247,  40.10702247,  11.05802247,  -4.13697753,  -1.00897753,
   -13.07497753,  10.61102247, -36.33497753, -62.23497753, -48.38097753,
    -9.49997753, -12.62797753, -15.30997753, -20.67297753, -13.96897753,
     2.11902247])

  e = wrap([ -1.94624719,  -3.43024719,  -0.29524719,  19.81475281,   5.51375281,
     5.96075281,  -1.63624719,   9.98275281,  -0.74224719,   3.72675281,
     4.61975281,  -2.08324719,   6.40775281,  -2.08324719,  -4.31824719,
   -10.57424719,  -8.34024719,  -7.44624719, -12.36224719, -11.46824719,
    39.47875281, -14.15024719, -43.19924719, -11.02124719,   7.30175281,
    -8.34024719,  -4.76524719,   2.83275281,   1.04475281,  -1.63624719,
     4.99275281,  27.85975281,  11.77075281,   8.19575281,   4.61975281,
    -2.53024719,  -2.53024719,   0.59775281, -11.91524719,   5.51375281])

  f = wrap([  1.76851685,   9.81251685,  -4.48848315,   1.76851685,   5.34351685,
    29.92351685,  21.66951685,  20.09151685,  30.27351685,   8.93051685,
     7.30175281,   3.27975281,  -8.78724719, -10.12724719, -12.80924719])

  g = wrap([ -9.4257191 ,  -15.8327191,   -5.3397191,   14.2462809,   15.1402809,  11.1182809,
     8.67451685,   9.15451685,  12.16751685, -13.42648315,  -0.29148315,
    15.31551685, -42.02848315,   7.21951685,   3.55651685, -13.87348315,
     4.45051685,  -2.06048315, -13.71448315,  -5.82848315,   0.87451685,
     5.3082809 ,   15.1402809,   13.7992809,   11.5652809,    0.3922809,   1.2862809,
     0.42751685,  -3.59448315,  -4.48848315,   2.21551685,  -3.59448315,
     7.9892809 ,    4.4142809,    8.8832809,    0.8392809,    0.8392809,   6.2022809,
    25.00751685,   7.75151685,   8.65351685,  -1.80648315,  -2.33248315,
     5.3082809 ,    3.9672809,    6.2022809,    3.9672809,   11.1182809,  -0.5017191])

  g = np.tile(g, 10)

  h = wrap([ -6.5888427 ,    5.9241573,   -4.8018427,   -3.4608427,    0.1141573,   6.3711573,
    -3.59448315,  11.15551685,   0.87451685,  -9.40448315,   5.79051685,
    16.0342809 ,   12.4582809,   22.2902809,   -1.3957191,   -5.8647191,  -0.5017191,
   -17.4837191 ,  -25.9757191,  -27.7627191,  -24.6347191,   -2.2897191, -16.5907191,
   -25.5287191 ,  -11.2277191,  -13.9087191,  -17.4837191,  -11.6747191, -23.7407191,
   -10.12724719, -23.98224719])

  i = wrap([-3.9078427 , -3.4608427,  -2.1198427,   3.2431573,  -3.9078427,
  -3.0138427, 21.16438202,  19.74038202,  40.06538202,  68.66738202,  -9.98861798,
     6.68451685,  -7.61648315, -24.59848315])

  # NUM_RUNS = 1
  NUM_RUNS = 9999
  res = 0.0
  for k in range(NUM_RUNS):
    res += var_red_fn(a);
    res += var_red_fn(b);
    res += var_red_fn(c);
    res += var_red_fn(d);
    res += var_red_fn(e);
    res += var_red_fn(f);
    res += var_red_fn(g);
    res += var_red_fn(h);
    res += var_red_fn(i);

  print res


# AS TAKEN from backend/train.py
def var_red_python(arr):
    # print arr
    # Computes a single term in the formular at
    # http://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction

    # import pdb
    # pdb.set_trace()
    N = float(len(arr))

    if N == 0.0:
        return 0.0

    n = np.tile(arr, N).reshape(-1, N)
    return 0.5 * (1.0/(N)) * np.sum((n - n.T)**2)


def var_red_c(arr):
  return var_red_lib_fn(ctypes.c_int(len(arr)), ctypes.c_void_p(arr.ctypes.data))

if __name__ == '__main__':


  #  ->  performance git:(master) x time python var_red.py
  #  1094714187.49
  #  python var_red.py  2.61s user 0.04s system 99% cpu 2.657 total
  #  ->  performance git:(master) x time python var_red.py
  #  1094714187.49
  #  python var_red.py  2.28s user 0.04s system 99% cpu 2.329 total
  #  ->  performance git:(master) x time python var_red.py
  #  1094714187.49
  #  python var_red.py  2.67s user 0.05s system 99% cpu 2.732 total

  # run_test(var_red_python)

  run_test(var_red_c)

  print '41849288.6171 << rougly expected result'
