# Notes on Scipy Transforms
1. DCT
    i. There are 8 types of DCT
    ii. Scipy implements Type2 DCT as dct(x), Type3 DCT as idct(x)
    iii. a DCT is a Fourier-related transform similar to the discrete Fourier transform (DFT), but using only real numbers

2. More concepts
    i. [1, N] x [N, N] = [1, N]
    ii. [N, N] -> DCT matrix

3. DCT2
    i. Assume x = [0,1,2,3,4]
    ii. Xk = [0 x cos(pi/2 x 1/2 x k)] + [1 x cos(pi/2 x 3/2 x k)] + [2 x cos(pi/2 x 5/2 x k)] + [3 x cos(pi/2 x 7/2 x k)] + [4 x cos(pi/2 x 9/2 x k)] 