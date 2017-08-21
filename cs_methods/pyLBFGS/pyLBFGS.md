# Notes on Scipy Transforms
1. DCT
    i. There are 8 types of DCT
    ii. Scipy implements Type2 DCT as dct(x), Type3 DCT as idct(x)
    iii. a DCT is a Fourier-related transform similar to the discrete Fourier transform (DFT), but using only real numbers

2. More concepts
    i. [1, N] x [N, N] = [1, N]
    ii. [N, N] -> DCT matrix