
# coding: utf-8

# In[1]:

import numpy as np
from PIL import Image # Part of the standard Python Library
import scipy.ndimage as scimg
from scipy.misc import imresize
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
np.set_printoptions(precision=4)


# # Links
# [CDF Gist](https://gist.github.com/CoderSherlock/834e9eb918eeb0dfee5f4550077f57f8)

# In[16]:

'''
2D CDF 9/7 Wavelet Forward and Inverse Transform (lifting implementation)
This code is provided "as is" and is given for educational purposes.
2008 - Kris Olhovsky - code.inquiries@olhovsky.com
'''

class cdf97():
    def __init__(self):
        self.nlevels = 1
        self.verbose = 0

    def fwt97_2d(self, img, nlevels=1):
        ''' Perform the CDF 9/7 transform on a 2D matrix signal m.
            nlevel is the desired number of times to recursively transform the
            signal. '''
        
        w = int(len(img[0]))
        h = len(img)
        
        # img_coeffs = np.array(img)
        img_coeffs = list(img)
        for i in range(nlevels):
            img_coeffs = self.fwt97(img_coeffs, w, h) # cols
            img_coeffs = self.fwt97(img_coeffs, w, h) # rows
            w = int(w/2)
            h = int(h/2)

        return img_coeffs

    def iwt97_2d(self, img_coeffs, nlevels=1):
        ''' Inverse CDF 9/7 transform on a 2D matrix signal m.
            nlevels must be the same as the nlevels used to perform the fwt.
        '''
        w = len(img_coeffs[0])
        h = len(img_coeffs)

        # Find starting size of m:
        for i in range(nlevels-1):
            h /= 2
            w /= 2

        img = list(img_coeffs)
        for i in range(nlevels):
            img = self.iwt97(img, w, h) # rows
            img = self.iwt97(img, w, h) # cols
            h *= 2
            w *= 2

        return img

    def fwt97(self, x, width, height):
        ''' Forward Cohen-Daubechies-Feauveau 9 tap / 7 tap wavelet transform
        performed on all columns of the 2D n*n matrix signal s via lifting.
        The returned result is s, the modified input matrix.
        The highpass and lowpass results are stored on the left half and right
        half of s respectively, after the matrix is transposed. '''
        # 9/7 Coefficients:
        a1 = -1.586134342
        a2 = -0.05298011854
        a3 = 0.8829110762
        a4 = 0.4435068522

        # Scale coeff:
        k1 = 0.81289306611596146 # 1/1.230174104914
        k2 = 0.61508705245700002 # 1.230174104914/2
        # Another k used by P. Getreuer is 1.1496043988602418
        
        x_coeffs = list(x)
        
        for col in range(width): # Do the 1D transform on all cols:
            ''' Core 1D lifting process in this loop. '''
            ''' Lifting is done on the cols. '''

            # Predict 1. y1
            for row in range(1, height-1, 2):
                x_coeffs[row][col] += a1 * (x_coeffs[row-1][col] + x_coeffs[row+1][col])
            x_coeffs[height-1][col] += 2 * a1 * x_coeffs[height-2][col] # Symmetric extension

            # Update 1. y0
            for row in range(2, height, 2):
                x_coeffs[row][col] += a2 * (x_coeffs[row-1][col] + x_coeffs[row+1][col])
            x_coeffs[0][col] +=  2 * a2 * x_coeffs[1][col] # Symmetric extension

            # Predict 2.
            for row in range(1, height-1, 2):
                x_coeffs[row][col] += a3 * (x_coeffs[row-1][col] + x_coeffs[row+1][col])
            x_coeffs[height-1][col] += 2 * a3 * x_coeffs[height-2][col]

            # Update 2.
            for row in range(2, height, 2):
                x_coeffs[row][col] += a4 * (x_coeffs[row-1][col] + x_coeffs[row+1][col])
            x_coeffs[0][col] += 2 * a4 * x_coeffs[1][col]

        # de-interleave
        if self.verbose : print ('Height:', height,' Width:', width)
        temp_bank = [[0]*width for i in range(height)]
        for row in range(height):
            for col in range(width):
                # k1 and k2 scale the vals
                # simultaneously transpose the matrix when deinterleaving
                # print ('Col:', col, ' Row:', row,' Row/2', row/2)
                if row % 2 == 0: # even
                    # print ('k1 * s[row][col]:', k1 * s[row][col])
                    temp_bank[col][int(row/2)] = k1 * x_coeffs[row][col]
                else:            # odd
                    # print ('k2 * s[row][col]:', k2 * s[row][col])
                    temp_bank[col][int(row/2 + height/2)] = k2 * x_coeffs[row][col]

        # write temp_bank to s:
        for row in range(width):
            for col in range(height):
                x_coeffs[row][col] = temp_bank[row][col]

        return x_coeffs

    def iwt97(self, x, width, height):
        ''' Inverse CDF 9/7. '''
        width = int(width)
        height = int(height)

        # 9/7 inverse coefficients:
        a1 = 1.586134342
        a2 = 0.05298011854
        a3 = -0.8829110762
        a4 = -0.4435068522

        # Inverse scale coeffs:
        k1 = 1.230174104914
        k2 = 1.6257861322319229

        # Interleave:
        if self.verbose : print ('Width:', width, ' Height:', height)
        s = list(x)
        temp_bank = [[0]*width for i in range(height)]
        for col in range(int(width/2)):
            for row in range(height):
                # k1 and k2 scale the vals
                # simultaneously transpose the matrix when interleaving
                temp_bank[col * 2][row] = k1 * s[row][col]
                temp_bank[col * 2 + 1][row] = k2 * s[row][int(col + width/2)]

        # write temp_bank to s:
        for row in range(width):
            for col in range(height):
                s[row][col] = temp_bank[row][col]

        for col in range(width): # Do the 1D transform on all cols:
            ''' Perform the inverse 1D transform. '''

            # Inverse update 2.
            for row in range(2, height, 2):
                s[row][col] += a4 * (s[row-1][col] + s[row+1][col])
            s[0][col] += 2 * a4 * s[1][col]

            # Inverse predict 2.
            for row in range(1, height-1, 2):
                s[row][col] += a3 * (s[row-1][col] + s[row+1][col])
            s[height-1][col] += 2 * a3 * s[height-2][col]

            # Inverse update 1.
            for row in range(2, height, 2):
                s[row][col] += a2 * (s[row-1][col] + s[row+1][col])
            s[0][col] +=  2 * a2 * s[1][col] # Symmetric extension

            # Inverse predict 1.
            for row in range(1, height-1, 2):
                s[row][col] += a1 * (s[row-1][col] + s[row+1][col])
            s[height-1][col] += 2 * a1 * s[height-2][col] # Symmetric extension

        return s

    def seq_to_img(self, m, pix):
        ''' Copy matrix m to pixel buffer pix.
        Assumes m has the same number of rows and cols as pix. '''
        for row in range(len(m)):
            for col in range(len(m[row])):
                pix[col,row] = m[row][col]
            

def gist_technique():
    # Load image.
    im = Image.open("test1_512.png") # Must be a single band image! (grey)

    # Create an image buffer object for fast access.
    pix = im.load()

    # Convert the 2d image to a 1d sequence:
    m = list(im.getdata())

    # Convert the 1d sequence to a 2d matrix.
    # Each sublist represents a row. Access is done via m[row][col].
    m = [m[i:i+im.size[0]] for i in range(0, len(m), im.size[0])]

    # Cast every item in the list to a float:
    for row in range(0, len(m)):
        for col in range(0, len(m[0])):
            m[row][col] = float(m[row][col])

    # Perform a forward CDF 9/7 transform on the image:
    m = fwt97_2d(m, 3)

    seq_to_img(m, pix) # Convert the list of lists matrix to an image.
    im.save("test1_512_fwt.png") # Save the transformed image.

    # Perform an inverse transform:
    m = iwt97_2d(m, 3)

    seq_to_img(m, pix) # Convert the inverse list of lists matrix to an image.
    im.save("test1_512_iwt.png") # Save the inverse transformation.

def me_technique(nlevels = 1, nx = 512, ny = 512):   
    
    url = "../../data/raw_data_cats/cat.1000.jpg"
    # url = "../../data/raw_data_cats/cat.1019.jpg"
    url = '../../noniterative/data/original/SRCNN/Test/Set14/baboon.bmp'
    
    img = scimg.imread(url, flatten=True)
    im_resize = imresize(img, (nx,ny)) #/255.0
    print ('0. Image:', im_resize.shape)
    f, axarr = plt.subplots(1,3, figsize = (10,10))
    axarr[0].imshow(im_resize, cmap = 'gray')
    
    # Step0 : Class Definition
    fwt97_obj  = cdf97()
    
    # Step1 : Perform a forward CDF 9/7 transform on the image:
    print ('\n1. Performing Forward CDF transform....')
    im_resize = im_resize.tolist()
    im_transform = fwt97_obj.fwt97_2d(im_resize, nlevels=nlevels)
    axarr[1].imshow(im_transform, cmap = 'gray')
    
    # Step2 : Perform an inverse CDF 9/7 transform on the image:
    print ('\n2. Performing Inverse CDF transform....')
    im_recon = fwt97_obj.iwt97_2d(im_transform, nlevels=nlevels)
    axarr[2].imshow(im_recon, cmap = 'gray')   
    
    # Step3 : Result
    pSNR = compare_psnr(np.array(im_resize).astype('float32'), np.array(im_recon).astype('float32') , dynamic_range=255)
    print ('pSNR:', pSNR)
    
    print ('im_resize:', id(im_resize), ' im_recon:', id(im_recon))

if __name__ == "__main__":
    pass
    # gist_technique()
    me_technique(nlevels=3)


# In[25]:

class cdf97_matlab():
    
    def filter9_7(self, verbose = 1):
        from scipy.signal import deconvolve
        
        p0 = np.array([-5, 0, 49, 0, -245, 0, 1225, 2048, 1225, 0, -245, 0, 49, 0, -5]) / 2048; #half-band filter
        # r = np.roots(p0)
        b0 = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1]);
        q0 = deconvolve(p0,b0)[0]
        
        r = np.vstack( (np.roots(q0).reshape((6,1)), -np.ones((8,1))) )
        
        h0_ip = []
        for each in r[1:5]:
            h0_ip.append(each[0])
        h0_ip.extend([-1, -1, -1, -1])
        h0_ip = np.array(h0_ip)
        if verbose : print ('h0_ip:',h0_ip)
        
        f0_ip = []
        f0_ip.append(r[0][0])
        f0_ip.append(r[5][0])
        f0_ip.extend([-1, -1, -1, -1])
        f0_ip = np.real(np.array(f0_ip))
        if verbose : print ('f0_ip:',f0_ip)
        
        h0 = np.poly(h0_ip)
        f0 = np.poly(f0_ip)
        
        if verbose : 
            print ('\nh0:',h0)
            print ('f0:',f0) 
                      
        h0 = np.real(np.sqrt(2) * h0/sum(h0));
        f0 = np.real(np.sqrt(2) * f0/sum(f0));
        
        if verbose : 
            print ('\nh0:',h0)
            print ('f0:',f0)
        
        f1 = h0 * [(-1) ** i for i in range(0, len(h0))]
        h1 = f0 * [(-1) ** i for i in range(1, len(f0)+1)]
        if verbose : 
            print ('\nh1:',h1)
            print ('f1:',f1)
            
        return h0,h1,f0,f1
    
    def idwt2d(self, x, f0, f1, mode = 1, maxlevel = 1, show = 0, verbose = 0):
        # rx=idwt2d(x,f0,f1,maxlevel);
        # Inverse 2D Discrete Wavelet Transform
        m,n = x.shape
        if verbose : print ('[Reconstrution] Original Image Size:', m, n)
            
        rx = np.zeros((m,n))
        m = int(m / (2 ** maxlevel))
        n = int(n / (2 ** maxlevel))
        if show:
            f, axarr = plt.subplots(1, maxlevel, figsize=(16,16))
        
        x_recon = np.array(x)
        for i in range(maxlevel):            
            if verbose : 
                print ('---------------------------> [Reconstrution] m,n:', m, n)
            xl = self.lphrec(x_recon[0:m, 0:n].T   , x_recon[0:m, n:2*n].T  , f0, f1, mode = mode, verbose = verbose)
            xh = self.lphrec(x_recon[m:2*m, 0:n].T , x_recon[m:2*m, n:2*n].T, f0, f1, mode = mode, verbose = verbose)
            x_recon[0:2*m,0:2*n] = self.lphrec(xl.T,xh.T,f0,f1, mode = mode, verbose = verbose)
            m = int(m*2)
            n = int(n*2)
            if show:
                axarr[i].imshow(x_recon, cmap = 'gray')
    
        return x_recon
    
    def lphrec(self, xl, xh, f0, f1, mode = 1, verbose = 0):
        # x = lphrec(xl,xh,f0,f1,mode)
            # Reconstruction of given scaling and wavelet components into the higher level 
            # scaling coeffs. using a LP PR synthesis bank
            #	Input:
            #		xl  : input scaling coeffs.
            #		      (vector [1,x_len/2] or list of vectors [m,x_len/2])
            #		xh  : input wavelet coeffs.
            #		      (vector [1,x_len/2] or list of vectors [m,x_len/2])
            #		f0  : synthesis filter coeffs. for F0(z) (vector [1,h0_len])
            #		f1  : synthesis filter coeffs. for F1(z) (vector [1,h1_len])
            #		mode: Extension Mode:
            #			0 - zero extension
            #			1 - symmetric extension
            #			2 - circular convolution
            #	Output:
            #		x   : scaling coeffs. at next higher level
            #		      (vector [1,x_len] or list of vectors [m,x_len])

        m, xl_len = xl.shape;
        h0_len = len(f0);
        h1_len = len(f1);
        if h0_len % 2 != h1_len % 2 :
            print('error: filter lengths must be EE or OO!');
        if (m,xl_len) != xh.shape:
            print('error: both channels must have same size!');

        x_len = int(xl_len*2)
        tb = int(h0_len % 2)
        ta = 1-tb
        
        xl2 = np.zeros((m,x_len))
        xh2 = np.zeros((m,x_len))
        
        temp_cols_xl2 = np.arange(2-tb-1, x_len, 2)
        temp_cols_xh2 = np.arange(2-1, x_len, 2)
            
        if verbose:
            print ('[Recon][lphrec] x_len, tb, ta :', x_len, tb, ta)
            print ('xl:', xl.shape, ' xl2:', xl2.shape)
            print ('xh:', xh.shape, ' xh2:', xh2.shape)
            print ('2-tb-1:2:x_len', 2-tb-1, 2, x_len)
            print ('temp_cols_xl2:', temp_cols_xl2.shape)
            print ('temp_cols_xh2:', temp_cols_xh2.shape)
        
        xl2[:,temp_cols_xl2] = xl;
        xh2[:,temp_cols_xh2] = xh;
        
        ext = int(np.floor(max(h0_len,h1_len)/2))

        if mode == 1:
            if tb == 1:
                xl2 = np.hstack((np.fliplr(xl2[:,1:ext+1+1]), xl2, np.fliplr(xl2[:,x_len-ext-1:x_len-1+1])));
                xh2 = np.hstack((np.fliplr(xh2[:,1:ext+1+1]), xh2, np.fliplr(xh2[:,x_len-ext-1:x_len-1+1])));
            else:
                xl2 = np.hstack((np.fliplr(xl2[:,1:ext+1])  , xl2 , np.zeros(m,1), np.fliplr(xl2[:,x_len-ext+1-1:x_len+1])))
                xh2 = np.hstack((-np.fliplr(xh2[:,1:ext+1]) , xh2 , np.zeros(m,1), -np.fliplr(xh2[:,x_len-ext+1-1:x_len+1])))
        elif mode == 2:
            xl2 = [xl2[:,x_len-ext+1+ta:x_len], xl2, xl2[:,1:ext+ta]];
            xh2 = [xh2[:,x_len-ext+1+ta:x_len], xh2, xh2[:,1:ext+ta]];

        else:
            xl2 = np.hstack((np.zeros((m,ext-ta)), xl2, np.zeros((m,ext+ta))));
            xh2 = np.hstack((np.zeros((m,ext-ta)), xh2, np.zeros((m,ext+ta))));

        x = np.zeros((m,x_len));
        f0_ord = int(h0_len-1)
        f1_ord = int(h1_len-1)
        s = int((f0_ord-f1_ord) / 2)
        if s >= 0:
            for i in range(0, x_len):
                x[:,i] = np.matmul(xl2[:,i:i+f0_ord+1],f0) + np.matmul(xh2[:,i+s:i+s+f1_ord+1], f1)
        else:
            for i in range(0, x_len):
                x[:,i] = np.matmul(xl2[:,i-s:i-s+f0_ord+1],f0) + np.matmul(xh2[:,i:i+f1_ord+1],f1)
        
        return x

    def dwt2d(self, x, h0, h1, mode = 1, maxlevel = 1, show=0):
        # Forward 2D Discrete Wavelet Transform
        if show == 1:
            f1, axarr1 = plt.subplots(1, maxlevel, figsize=(16,16))
        m, n = x.shape
        x_coeffs = np.array(x)
        
        for i in range(maxlevel):
            [xl,xh]    = self.lphdec(x_coeffs[0:m,0:n], h0, h1, mode = mode)
            [xll,xlh]  = self.lphdec(xl.T, h0, h1, mode = mode);
            [xhl,xhh]  = self.lphdec(xh.T, h0, h1, mode = mode);
            x_coeffs[0:m,0:n] = np.vstack( ( np.hstack((xll,xhl)), np.hstack((xlh,xhh))) ).T
            m = int(m/2)
            n = int(n/2)
            if show == 1:
                axarr1[i].imshow(x_coeffs, cmap='gray')
        
        
        return x_coeffs
    
    def lphdec(self, x, h0, h1, mode=1, verbose = 0):
        # [xl,xh] = lphdec(x,h0,h1,mode)
            # Decomposition of given scaling coefficients into their scaling and wavelet 
            # parts using a LP PR analysis bank
            #	Input:
            #		x   : input scaling coeffs. (vector [1,x_len] or 
            #		      matrix [m,x_len], i.e., list of m vectors [1,x_len])
            #		h0  : analysis filter coeffs. for H0(z) (vector [1,h0_len])
            #		h1  : analysis filter coeffs. for H1(z) (vector [1,h1_len])
            #		mode: Extension Mode:
            #			0 - zero extension 
            #			1 - symmetric extension
            #			2 - circular convolution
            #	Output:
            #		xl  : scaling coeffs. at next coarser level (low pass)
            #		      (vector [1,x_len/2] or matrix [m,x_len/2])
            #		xh  : wavelet coeffs. at next coarser level (high pass)
            #		      (vector [1,x_len/2] or matrix [m,x_len/2])

        (m,x_len) = x.shape;
        h0_len = len(h0);
        h1_len = len(h1);
        if h0_len % 2 != h1_len % 2:
            print ('error: filter lengths must be EE or OO!')

        ext = int(np.floor(max(h0_len,h1_len)/2)) # extension size
        tb = int(h0_len % 2)                       # change extension type for EE- or OO-FB
        if mode == 1:
            temp1 = np.fliplr(x[:,1+tb-1:ext+tb])
            temp2 = x
            temp3 = np.fliplr(x[:,x_len-ext+1-tb-1:x_len-tb])
            x = np.hstack((temp1, temp2, temp3))
            if verbose:
                print ('temp1:', temp1.shape, ' temp2:', temp2.shape, ' temp3:', temp3.shape)
                print ('x:', x.shape)

        elif mode == 2:
            x = [x[:,x_len-ext+1:x_len], x, x[:,1:ext]]
        else:
            x = np.hstack((np.zeros((m,ext)), x, np.zeros((m,ext))));

        lenn = int(np.floor((x_len+1)/2))
        xh = np.zeros((m,lenn))
        xl = np.zeros((m,lenn))
        s = (h0_len-h1_len)/2
        
        if s >= 0:
            k1 = int(2 - tb)
            k2 = int(2 + s)
        else:
            k2 = 2
            k1 = int(2 - tb - s)
        
        h0_ord = int(h0_len-1)
        h1_ord = int(h1_len-1)
        
        for i in range(lenn):
            if verbose:
                print ('i:',i)
                print (' x[:,k1:k1+h0_ord+1]:',x[:,k1:k1+h0_ord+1].shape, ' h0:', h0.shape)
                print ('x[:,k1:k1+h0_ord+1] * h0 ', (x[:,k1:k1+h0_ord+1] * h0).shape)
            if i == 0:
                pass
                # print (' x[:,k1:k1+h0_ord+1]:',x[:,k1:k1+h0_ord+1].shape, ' h0:', h0.shape)
                # print (' x[:,k1:k1+h0_ord+1]:',x[:,k1:k1+h0_ord+1][0:5, 0:5], ' h0:', h0)
                # print ('np.matmul(x[:,k1:k1+h0_ord+1], h0)', np.matmul(x[:,k1:k1+h0_ord+1], h0)[:5])
                # print (' x[:,k2:k2+h1_ord+1]:',x[:,k2:k2+h1_ord+1].shape, ' h1:', h1.shape)
                # print (' x[:,k2:k2+h1_ord+1]:',x[:,k2:k2+h1_ord+1][0:5, 0:5], ' h1:', h1)
                # print ('np.matmul(x[:,k2:k2+h1_ord+1], h1)', np.matmul(x[:,k2:k2+h1_ord+1], h1)[:5])
            xl[:,i] = np.matmul(x[:,k1-1:k1+h0_ord], h0)
            xh[:,i] = np.matmul(x[:,k2-1:k2+h1_ord], h1)
            k1 = int(k1 + 2)
            k2 = int(k2 + 2)
        
        return xl, xh

def cdf97_matlab_sample(url, nx, ny, L = 3, mode = 0, resized=False, compress=False):
    import numpy as np
    from scipy.ndimage import imread
    from scipy.misc import imresize
    from skimage.measure import compare_psnr
    import matplotlib.pyplot as plt
    
    get_ipython().magic('matplotlib inline')
    np.set_printoptions(precision=4)
    
    if url == '':
        url = '../../data/lena512.bmp'
        # url = '../../data/raw_data_cats/cat.12.jpg'
    
    x = imread(url, flatten = True)
    x_resize = imresize(x, (nx,ny)).astype(np.float16)
    
    m, n = x.shape
    f, axarr = plt.subplots(1,4, figsize=(10,10))
    print ('Original Image:', x.shape)
    print ('Reshaped Image:', x_resize.shape)
    axarr[0].imshow(x, cmap = 'gray')
    axarr[1].imshow(x_resize, cmap = 'gray')
    if resized:
        x = np.array(x_resize)
    
    cdf97_obj = cdf97_matlab()
    
    """ STEP 1 """
    h0,h1,f0,f1 = cdf97_obj.filter9_7(verbose = 0)
    # print ('h0,h1,f0,f1', h0.shape, h1.shape, f0.shape, f1.shape)
    
    """ STEP 2 """
    # L = int(np.floor(np.log2(m)) - 3)
    x_coeffs = cdf97_obj.dwt2d(x, h0, h1, mode = mode, maxlevel = L, show = 1)
    axarr[2].imshow(x_coeffs, cmap = 'gray')
    if compress:
        compres_thresh = 10
        print ('Total pixels:', x_coeffs.shape[0] * x_coeffs.shape[1], ' Non Zero Pixels:', np.count_nonzero(x_coeffs))
        x_coeffs[(x_coeffs < compres_thresh) & (x_coeffs > -1*compres_thresh)] = 0
        print ('Total pixels:', x_coeffs.shape[0] * x_coeffs.shape[1], ' Non Zero Pixels:', np.count_nonzero(x_coeffs))
    
    """ STEP 3 """
    x_recon = cdf97_obj.idwt2d(x_coeffs, f0, f1,mode = mode, maxlevel = L, show = 1, verbose = 0)
    axarr[3].imshow(x_recon, cmap='gray')
    
    pSNR = compare_psnr(x.astype(np.float32), x_recon.astype(np.float32), dynamic_range=255)
    print ('pSNR (of transform basis):', pSNR)

if __name__ == "__main__":
    pass
#     nx = ny = 512
#     url = '../../noniterative/data/original/SRCNN/Test/Set14/baboon.bmp'
#     cdf97_matlab_sample(url, nx, ny, L = 3, mode = 0, resized = True, compress = False) 


# In[ ]:



