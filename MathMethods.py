import numpy as np
from scipy import ndimage as img

# convolution filter for laplacian
LaplassianFltr = np.array([
    [0, 1, 0],
    [1,-4, 1],
    [0, 1, 0]
])

DeriviativeFltr = np.array([
    [ 0, 0, 0],
    [-1, 0, 1],
    [ 0, 0, 0]
])

X = 0
Y = 1

#make array of relative adjacent positions
Adjacent = np.array([ [-1,0], [1,0], [0,1], [0,-1] ])

#first partial derivative of one of the array axis
def Grad(target, axis, dx=1):
    answer = np.ndarray(target.shape)
    
    
    itr = np.nditer(target, flags=["multi_index"])
    for i in itr:
        index = np.asarray(itr.multi_index)
        
        back_idx = np.copy(index)
        back_idx[axis] -= 1
        fwrd_idx = np.copy(index)
        fwrd_idx[axis] +=1
        
        
        back_value = target[tuple(back_idx)] if (back_idx >= 0).all() else 0
        fwrd_value = target[tuple(fwrd_idx)] if (fwrd_idx < target.shape).all() else 0
        
        answer[tuple(index)] = (fwrd_value - back_value)/dx
        
    return answer


#second partial derivative of one of the array axis        
def Grad2(target, axis, dx=1):
    answer = np.ndarray(target.shape)
    
    
    itr = np.nditer(target, flags=["multi_index"])
    for i in itr:
        index = np.asarray(itr.multi_index)
        
        back_idx = np.copy(index)
        back_idx[axis] -= 1
        fwrd_idx = np.copy(index)
        fwrd_idx[axis] +=1

        
        back_value = target[tuple(back_idx)] if (back_idx >= 0).all() else 0
        fwrd_value = target[tuple(fwrd_idx)] if (fwrd_idx < target.shape).all() else 0
        
        answer[tuple(index)] = (fwrd_value - 2*i  +  back_value)/dx**2
        
    return answer

def Gradient2d(U):
    Xcomp = img.convolve(U, DeriviativeFltr)
    YComp = img.convolve(U, DeriviativeFltr.transpose())

    return np.array([Xcomp, YComp]).transpose(1,2,0)


def Divergence2d(U):
    return Grad(U[:,:,X], X) + Grad(U[:,:,Y], Y)


def Generate_Laplassian_Mtx(size):

    I = np.identity(size)
    D = np.diag([-4]*size) + np.diag([1]*(size-1), 1) + np.diag([1]*(size-1), -1)

    LapTrans = np.zeros((size,)*4)
    LapTrans[range(size), range(size),:,:] = D
    LapTrans[range(size-1), range(1,size),:,:] = I
    LapTrans[range(1,size), range(size-1),:,:] = I
    LapTrans = [[ j for j in i ] for i in LapTrans]
    LapTrans = np.block(LapTrans)

    return LapTrans
    

def Laplassian2d(U, dx=1, mode='constant'):
    return img.convolve(U, LaplassianFltr, mode=mode)/dx**2

## TODO incorperate dx into inv laplassian functions

def Inverse_Laplassian2d_Sq(U, mode='constant', dx=1):
    assert(type(U) == np.ndarray)
    assert(U.ndim == 2)
    assert(U.shape[0] == U.shape[1])
    
    Padded = np.pad(U, 1, mode=mode)

    dims = Padded.shape
    UVec = Padded.flatten()
    Len = dims[0]

    LapTrans = Generate_Laplassian_Mtx(Len)

    Ans = np.linalg.solve(LapTrans, UVec)
    Ans = Ans.reshape(dims)

    Ans = Ans[1:-1,1:-1]

    return Ans


# https://people.eecs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html
# Successive Overrelaxation (SOR)
# first time citing a source in my code
def Inverse_Laplassian2d_SOR(U, err=1e-5, max_iter=200, init=np.NaN, dx=1):
    if (not np.isnan(init).all()):
        assert(init.shape == U.shape)
    assert(U.ndim == 2)
    

    dim = np.array(U.shape)
    size = max(U.shape)

    # w is size of step in the direction of answer
    w_opt = 2 / ( 1 + np.sin( np.pi / (1+size) ) )
    #w_opt = 1

    # make filters to swap between checkboard sections
    FilterBase = np.array([ [i % dim[0], i // dim[1]] for i in range(U.size) ])
    Checker = (FilterBase.sum(1))%2
    FilterBlack = FilterBase[ Checker == 0 ]
    FilterRed   = FilterBase[ Checker == 1 ]


    #init interation values
    if (np.isnan(init).all()):
        A_old = np.zeros(dim)
    else:
        A_old = init
    A = np.zeros(dim)

    for i in range(max_iter):
        #scoot A fields back to make space for new Iter
        A = np.zeros(dim)

        # use old values on black squares of the checkerboard
        for j in FilterBlack:
            ind = tuple(j)
            AroundVals = [ A_old[tuple(j+d)] if ( ([0,0] <= j+d) & (j+d < dim) ).all() else A_old[ind] for d in Adjacent]
            Estimate = ( np.sum(AroundVals) - dx*dx*U[ind] )/4
            Direction = Estimate - A_old[ind]

            A[ind] = A_old[ind] + ( w_opt * Direction )

        #use the values just make on the black square to make values for red squares
        for j in FilterRed:
            ind = tuple(j)
            AroundVals = [ A[tuple(j+d)] if ( ([0,0] <= j+d) & (j+d < dim) ).all() else A_old[ind] for d in Adjacent]
            Estimate = ( np.sum(AroundVals) - dx*dx*U[ind] )/4
            Direction = Estimate - A_old[ind]

            A[ind] = A_old[ind] + ( w_opt * Direction )
        
        #end loop if done convergine
        if (i%20==0):
            if (abs(A_old-A) < err).all():
                break

        A_old = np.copy(A)

    return A

