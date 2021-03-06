"""Gradient Projection for Sparse Reconstruction.
Solves the convex problem 

.. math::

    arg min_x = 0.5*|| y - A x ||_2^2 + \tau || x ||_1, 

where :math:`A` is a generic matrix. 

For further details about the TwIST algorithm, see the paper:

.. [1]
    "Gradient Projection for Sparse Reconstruction: Application
    to Compressed Sensing and Other Inverse Problems"
    by Mario A. T. Figueiredo, Robert D. Nowak, Stephen J. Wright,
    Journal of Selected Topics on Signal Processing, December 2007

The python implementation is based on the Matlab implemebtation
of GPSR.
Copyright (2007): Jario Figueiredo, Robert Nowak, Stephen Wright

GPSR is distributed under the terms of 
the GNU General Public License 2.0.

Permission to use, copy, modify, and distribute this software for
any purpose without fee is hereby granted, provided that this entire
notice is included in all copies of any software which is or includes
a copy or modification of this software and in all copies of the
supporting documentation for such software.
This software is being provided "as is", without any express or
implied warranty.  In particular, the authors do not make any
representation or warranty of any kind concerning the merchantability
of this software or its fitness for any particular purpose.

.. codeauthor:: Amit Aides <amitibo@tx.technion.ac.il>

"""

from __future__ import division
import numpy as np
import time

EPS = np.finfo(float).eps


def GPSR_raw(
    y,
    A,
    tau,
    psi_function=None,
    phi_function=None,
    lam1=1e-4,
    alpha=0,
    beta=0,
    AT=None,
    stop_criterion=3,
    tolA=0.01,
    debias=False,
    tolD=0.0001,
    maxiter=10000,
    miniter=5,
    maxiter_debias=200,
    miniter_debias=5,
    init=0,
    continuation=0,
    continuation_steps=5,
    first_tau_factor=None,
    true_x=None,
    verbose=True
    ):
    """Gradient Projection for Sparse Reconstruction
    Solves the convex problem 
    
    .. math::
    
        arg min_x = 0.5*|| y - A x ||_2^2 + \tau || x ||_1, 
    
    where :math:`A` is a generic matrix. 
    
    For further details about the TwIST algorithm, see the paper:
    
    .. [1]
        "Gradient Projection for Sparse Reconstruction: Application
        to Compressed Sensing and Other Inverse Problems"
        by Mario A. T. Figueiredo, Robert D. Nowak, Stephen J. Wright,
        Journal of Selected Topics on Signal Processing, December 2007
    
    Parameters
    ----------
    
    y : array,
       1D vector or 2D array (image) of observations.
        
    A : {array, function handle},
        if y and x are both 1D vectors, :math:`A` can be a 
        k*n (where k is the size of y and n the size of x)
        matrix or a handle to a function that computes
        products of the form :math:`Av`, for some vector v.
        In any other case (if y and/or x are 2D arrays), 
        A has to be passed as a handle to a function which computes 
        products of the form :math:`Ax`; another handle to a function 
        AT which computes products of the form :math:`A^Tx` is also required 
        in this case. The size of x is determined as the size
        of the result of applying AT.
   
    tau : float,
        regularization parameter, usually a non-negative real
        parameter of the objective function (see above).
     
    AT : function handle, optional
        function that implements the multiplication by the conjugate
        of A, when A is a function handle. If A is an array, AT is ignored.
    
    stop_criterion : {0, 1, 2, 3}, optional (default=3)
        type of stopping criterion to use
            0 = algorithm stops when the relative 
                change in the number of non-zero 
                components of the estimate falls 
                below tolA
            1 = stop when the relative 
                change in the objective function 
                falls below tolA
            2 = stop when the relative norm of the difference between 
                two consecutive estimates falls below toleranceA
            3 = stop when the objective function 
                becomes equal or less than toleranceA.
    
    tolA : float, optional (default=0.01)
        stopping threshold.
    
    debias : bool, optional (default=False)
        debiasing option
        
        .. note:: Debiasing is an operation aimed at the 
            computing the solution of the LS problem 

            .. math::
              
                arg min_x = 0.5*|| y - A^T x ||_2^2 
    
            where ::math:`A^T` is the  submatrix of A obtained by
            deleting the columns of A corresponding to components
            of x set to zero by the GPSR algorithm
                    
    
    tolD : float, optional (default=0.0001)
        stopping threshold for the debiasing phase.
        If no debiasing takes place, this parameter, is ignored.
    
    maxiter : int, optional (default=10000)
        maximum number of iterations allowed in the
        main phase of the algorithm.
    
    miniter : int, optional (default=5)
        minimum number of iterations performed in the
        main phase of the algorithm.
    
    maxiter_debias : int, optional (default=5)
        maximum number of iterations allowed in the
        debising phase of the algorithm.
    
    miniter_debias : int, optional (default=5)
        minimum number of iterations to perform in the
        debiasing phase of the algorithm.
    
    init : {0, 1, 2, array}, optional (default=0)
        must be one of 
        * 0 -> Initialization at zero. 
        * 1 -> Random initialization.
        * 2 -> initialization with ::math:`A^Ty`.
        * array -> initialization provided by the user.
    
    enforce_monotone : bool, optional (default=True)
        enforce monotonic decrease in f. 
    
    sparse : bool, optional (default=True)
        Accelarates the convergence rate when the regularizer 
        :math:`\Phi(x)` is sparse inducing, such as ::math:`||x||_1`.
                  
    true_x : array, optional (default=None)
        if the true underlying x is passed in this argument,
        MSE evolution is computed
    
    Verbose : bool, optional (default=False)
        work silently (False) or verbosely (True) (default 
    

    Returns
    -------
    x : array,
        solution of the main algorithm
    
    x_debias : array,
        solution after the debiasing phase;
        if no debiasing phase took place, this
        variable is [].
    
    objective : array,
        sequence of values of the objective function
    
    times : arraym
        CPU time after each iteration
    
    debias_start : int,
        iteration number at which the debiasing 
        phase started. If no debiasing took place,
        this variable is returned as zero.
    
    mses : array,
        sequence of MSE values, with respect to true_x,
        if it was given; if it was not given, mses is [].
    
    max_svd : float,
        inverse of the scaling factor, determined by TwIST,
        applied to the direct operator (A/max_svd) such that
        every IST step is increasing.
    """
    
    #
    # maj_max_sv: majorizer for the maximum singular value of operator A
    #
    max_svd = 1

    #
    # Set the defaults for outputs that may not be computed
    #
    debias_start = 0
    x_debias = []
    mses = []
    
    #
    # twist parameters
    #                
    lamN = 1
    lam1 = 1e-4
    rho0 = (1-lam1/lamN) / (1+lam1/lamN)

    alpha=0
    beta=0
    
    if alpha == 0:
        alpha = 2 / (1 + np.sqrt(1-rho0**2.))
    
    if beta == 0:
        beta = alpha*2/(lam1+lamN)
    
    if stop_criterion not in range(4):
        raise Exception('Unknwon stopping criterion')
    
    #
    # if A is a function handle, we have to check presence of AT
    #
    if isFunction(A) and not isFunction(AT):
        raise Exception('The function handle for the transpose of A is missing')
    
    #
    # if A is a matrix, we find out dimensions of y and x,
    # and create function handles for multiplication by A and A',
    # so that the code below doesn't have to distinguish between
    # the handle/not-handle cases
    #
    if not isFunction(A):
        AT = lambda x: np.dot(A.T, x.reshape((-1, 1))).reshape(y.shape)
        A = lambda x: np.dot(A, x.reshape((-1, 1))).reshape(y.shape)
    
    #
    # from this point down, A and AT are always function handles.
    # Precompute A'*y since it'll be used a lot
    Aty = AT(y)

    #
    # If no psi and phi were given, simply use the l1 norm.
    #
    psi_soft = False
    if psi_function == None:
        psi_function = softThreshold
        psi_soft = True

    phi_l1 = False
    if phi_function == None:
        phi_function = lambda x: np.sum(np.abs(x))
        phi_l1 = True
        
    #--------------------------------------------------------------
    # Initialization
    #--------------------------------------------------------------
    if np.isscalar(init):
        if init == 0:
            x = AT(np.zeros(y.shape)) # initialize at zero, using AT to find the size of x
        elif init == 1:
            x = np.random.randn(AT(np.zeros(y.shape)).shape) # initialize randomly, using AT to find the size of x
        elif init == 2:
            x = Aty # initialize x0 = A'*y
        else:
            raise Exception("Unknown 'Initialization' option")
    else:
        if A(init).shape == y.shape:
            x = init
        else:
            raise Exception("Size of initial x is not compatible with A")

    # now check if tau is an array; if it is, it has to have the same size as x
    if not np.isscalar(tau) and tau.shape != x.shape:
        raise Exception('Parameter tau has wrong dimensions; it should be scalar or size(x)')

    # if the true x was given, check its size
    compute_mse = False
    plot_ISNR = False
    if true_x.any() != None:
        if true_x.size != x.size:
            raise Exception('Initial x has incompatible size')
        compute_mse = True
        plot_ISNR = True
    
    # if tau is large enough, in the case of phi = l1, thus psi = soft,the optimal solution is the zero vector
    if phi_l1:
        max_tau = np.max(np.abs(Aty))
        
        if tau >= max_tau and psi_soft:
            x = np.zeros(Aty.shape)
            objective = [0.5*np.sum(y * y)]
            times = [0]
            
            if compute_mse:
                mses.append(np.sum((true_x**2)))
            
            return x, x_debias, objective, times, debias_start, mses, max_svd

    # define the indicator vector or matrix of nonzeros in x
    nz_x = x != 0
    num_nz_x = np.sum(nz_x)

    # Compute and store initial value of the objective function
    resid = y - A(x)
    prev_f = 0.5*np.sum(resid * resid) + tau*phi_function(x)

    #
    # start the clock
    #
    t0 = time.time()
    times = [0]
    objective = [prev_f]
    
    if compute_mse:
        mses.append(np.sum((x-true_x)**2))
    
    iter = 1
    if verbose:
        print ('\nInitial objective = %10.6e,  nonzeros=%7d' % (prev_f, num_nz_x))
    
    #
    # variables controling first and second order iterations
    #
    IST_iters = 0
    TwIST_iters = 0
    
    #
    # initialize
    #
    xm2 = x # TwIST
    xm1 = x # IST

    #-------------------------------------------------------------------------------------------------------------------------------
    # TwIST iterations
    #-------------------------------------------------------------------------------------------------------------------------------
    cont_outer = True
    while cont_outer:
        #
        # gradient
        #
        grad = AT(resid)
        while True:
            #
            # IST estimate
            #
            x = psi_function(xm1 + grad/max_svd, tau/max_svd)      #xm1 = x, max_svd = 1, , psi_function = softThreshold(x, threshold); tau = regularization operator
            if (IST_iters >= 2) or (TwIST_iters != 0):
                #
                # set to zero the past when the present is zero
                # suitable for sparse inducing priors
                #
                if sparse:
                    xm1[x == 0] = 0
                    xm2[x == 0] = 0
                
                #
                # two-step iteration
                #
                xm2 = (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x  #alpha, beta = fixed constants  {constant1}*xm1 + {constant2}*xm2 + {constant2}*x
 
                #
                # compute residual
                #
                resid = y - A(xm2)
                f = 0.5*np.sum(resid *resid) + tau*phi_function(xm2) #f (or objective) = 1/2 * sum(error^2) + regul_operator * phi(xm2) ; phi = regularization function (l1 norm for this code)
                
                if (f > prev_f) and enforce_monotone:  #default is False. So ignore
                    # do a IST iteration if monotonocity fails
                    TwIST_iters = 0
                else:
                    # TwIST iterations
                    TwIST_iters += 1
                    IST_iters = 0
                    x = xm2
                    
                    if TwIST_iters % 10000 == 0:
                        max_svd *= 0.9
                    #
                    # break while loop
                    #
                    break

            else:
                resid = y-A(x)
                f = 0.5*np.sum(resid * resid) + tau*phi_function(x)
                         
                if f > prev_f:
                    #
                    # if monotonicity  fails here  is  because
                    # max eig (A'A) > 1. Thus, we increase our guess
                    # of max_svs
                    #
                    max_svd *= 2
                    
                    if verbose:
                        print ('Incrementing S=%2.2e' % max_svd)
                        
                    IST_iters = 0
                    TwIST_iters = 0
                    
                else:
                    TwIST_iters += 1
                    
                    #
                    # break while loop
                    #
                    break

        xm2 = xm1
        xm1 = x
        
        #
        # Update the number of nonzero components and its variation
        #
        nz_x_prev = nz_x
        nz_x = x != 0
        num_nz_x = np.sum(nz_x)
        num_changes_active = np.sum(nz_x != nz_x_prev)

        #
        # take no less than miniter and no more than maxiter iterations
        #
        if stop_criterion == 0:
            #
            # compute the stopping criterion based on the change
            # of the number of non-zero components of the estimate
            #
            criterion = num_changes_active
        elif stop_criterion == 1:
            #
            # compute the stopping criterion based on the relative
            # variation of the objective function.
            #
            criterion = np.abs(f-prev_f)/prev_f
        elif stop_criterion == 2:
            #
            # compute the stopping criterion based on the relative
            # variation of the estimate.
            #
            criterion = np.linalg.norm((x-xm1).ravel()) / np.linalg.norm(x.ravel())
        elif stop_criterion == 3:
            #
            # continue if not yet reached target value tolA
            #
            criterion = f;
        else:
            raise Exception('Unknwon stopping criterion');

        cont_outer = (iter <= maxiter) and (criterion > tolA)
        
        if iter <= miniter:
            cont_outer = True

        iter += 1
        prev_f = f
        objective.append(f)
        
        times.append(time.time()-t0)

        if compute_mse:
            err = true_x - x
            mses.append(np.sum(err*err))

        #
        # Print out the various stopping criteria
        #
        if verbose:
            if plot_ISNR:
                print ('Iteration=%4d, ISNR=%4.5e  objective=%9.5e, nz=%7d, criterion=%7.3e' % \
                      (iter, 10*np.log10(np.sum((y-A(true_x))**2)/np.sum((x-true_x)**2)), f, num_nz_x, criterion/tolA))
            else:
                print ('Iteration=%4d, objective=%9.5e, nz=%7d,  criterion=%7.3e' % (iter, f, num_nz_x, criterion/tolA))
    
    #
    #--------------------------------------------------------------
    # end of the main loop
    #--------------------------------------------------------------
    # Printout results
    #
    if verbose:
        print ('\nFinished the main algorithm!\nResults:')
        print ('||A x - y ||_2 = %10.3e' % np.sum(resid * resid))
        print ('||x||_1 = %10.3e' % np.sum(np.abs(x)))
        print ('Objective function = %10.3e' % f)
        print ('Number of non-zero components = %d' % num_nz_x)
        print ('CPU time so far = %10.3e' % times[-1])
    
    
    #
    #--------------------------------------------------------------
    # If the 'Debias' option is set to 1, we try to
    # remove the bias from the l1 penalty, by applying CG to the
    # least-squares problem obtained by omitting the l1 term
    # and fixing the zero coefficients at zero.
    #--------------------------------------------------------------
    #
    if debias:
        if verbose:
            print ('\nStarting the debiasing phase...\n')
       
        x_debias = x.copy()
        debias_start = iter
        
        #
        # calculate initial residual
        #
        resid = A(x_debias)
        resid = resid-y
        rvec = AT(resid)
        
        #
        # mask out the zeros
        #
        zeroind = x_debias == 0
        rvec[zeroind] = 0
        rTr_cg = np.sum(rvec * rvec)
        
        #
        # Set convergence threshold for the residual || RW x_debias - y ||_2
        #
        tol_debias = tolD * rTr_cg
        
        #
        # initialize pvec
        #
        pvec = -rvec
        
        #
        # main loop
        #
        cont_debias_cg = True
        while cont_debias_cg:
            #
            # calculate A*p = Wt * Rt * R * W * pvec
            #
            RWpvec = A(pvec)
            Apvec = AT(RWpvec)

            #
            # mask out the zero terms
            #
            Apvec[zeroind] = 0

            #
            # calculate alpha for CG
            #
            alpha_cg = rTr_cg / np.sum(pvec * Apvec)

            #
            # take the step
            #
            x_debias = x_debias + alpha_cg * pvec
            resid = resid + alpha_cg * RWpvec
            rvec  = rvec  + alpha_cg * Apvec

            rTr_cg_plus = np.sum(rvec * rvec)
            beta_cg = rTr_cg_plus / rTr_cg
            pvec = -rvec + beta_cg * pvec

            rTr_cg = rTr_cg_plus

            iter += 1

            objective.append(0.5*np.sum(resid * resid) + tau*phi_function(x_debias))
            times.appebd(time.time() - t0)

            if compute_mse:
                err = true_x - x_debias
                mses.append(np.sum(err * err))

            #
            # in the debiasing CG phase, always use convergence criterion
            # based on the residual (this is standard for CG)
            #
            if verbose:
                print (' Iter = %5d, debias resid = %13.8e, convergence = %8.3e\n' % \
                      (iter, np.sum(resid * resid), rTr_cg / tol_debias))
                
            cont_debias_cg = \
                (iter-debias_start <= miniter_debias ) or \
                ((rTr_cg > tol_debias) and (iter-debias_start <= maxiter_debias))

        if verbose:
            print ('\nFinished the debiasing phase!\nResults:')
            print ('||A x - y ||_2 = %10.3e' % np.sum(resid * resid))
            print ('||x||_1 = %10.3e' % np.sum(np.abs(x)))
            print ('Objective function = %10.3e' % f)
            
            nz = x_debias != 0
            print ('Number of non-zero components = %d' % np.sum(nz))
            print ('CPU time so far = %10.3e\n' % times[-1])

    if compute_mse:
        mses = np.array(mses) / true_x.size
    
    return x, x_debias, objective, times, debias_start, mses, max_svd
    

def isFunction(f):
    """
    Check if object is a function.
    """
    import types
    return isinstance(f, types.FunctionType) or isinstance(f, types.MethodType) or hasattr(f, '__call__')


def softThreshold(x, threshold):
    """
    Apply Soft Thresholding
    
    Parameters
    ----------
    
    x : array-like
        Vector to which the soft thresholding is applied
        
    threshold : float
        Threhold of the soft thresholding
    
    Returns:
    --------
    y : array
        Result of the applying soft thresholding to x.
        
        .. math::
              
            y = sign(x) \star \max(\abs(x)-threshold, 0)
    """
    
    y = np.abs(x) - threshold
    y[y<0] = 0
    y[x<0] = -y[x<0]
    
    return y


def hardThreshold(x, threshold):
    """
    Apply Hard Thresholding
    
    Parameters
    ----------
    
    x : array-like
        Vector to which the hard thresholding is applied
        
    threshold : float
        Threhold of the hard thresholding
    
    Returns:
    --------
    y : array
        Result of the applying hard thresholding to x.
        
        .. math::
              
            y = x * (\abs(x) > threshold)
    """
    
    y = np.zeros_like(x)
    ind = np.abs(x) > threshold
    y[ind] = x[ind]
    
    return y