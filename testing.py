import MathMethods as MM
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as img

TSize = 25
testVal = np.arange(TSize**2).reshape((TSize,TSize))

# https://people.eecs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html
# Successive Overrelaxation (SOR)
# first time citing a source in my code
def Inverse_Laplassian(U):
    assert(U.shape[0] == U.shape[1])
    

    dim = np.array(U.shape)
    size = U.shape[0]

    # w is size of step in the direction of answer
    w_opt = 2 / ( 1 + np.sin( np.pi / (1+size) ) )
    #w_opt = 1
    # error reduction per iter. err = rho^[iter] * err_0
    rho = ( np.cos( np.pi / (size+1) ) / ( 1 + np.sin( np.pi / (size+1) ) ) )**2

    #log base rho of 1e-4 with base change equation
    errorReduction = 1e-20
    iters = int( np.ceil( np.log(errorReduction) / np.log(rho) ) )
    #iters = 500

    # make filters to swap between checkboard sections
    FilterBase = np.array([ [i % size, i // size] for i in range(U.size) ])
    FilterBlack = np.array([ FilterBase[2*i] for i in range((U.size + 1)//2) ])
    FilterRed = np.array([ FilterBase[2*i+1] for i in range(U.size//2) ])

    #make array of relative adjacent positions
    Adjacent = np.array([ [-1,0], [1,0], [0,1], [0,-1] ])

    #init interation values
    A_old = np.zeros(dim)
    A = np.zeros(dim)

    for i in range(iters):
        #scoot A fields back to make space for new Iter
        A_old = np.copy(A)
        A = np.zeros(dim)

        if i > 40:
            pass

        # use old values on black squares of the checkerboard
        for j in FilterBlack:
            ind = tuple(j)
            AroundVals = [ A_old[tuple(j+d)] if ( ([0,0] <= j+d) & (j+d < dim) ).all() else A_old[ind] for d in Adjacent]
            Estimate = ( np.sum(AroundVals) - U[ind] )/4
            Direction = Estimate - A_old[ind]

            A[ind] = A_old[ind] + w_opt * Direction
            test = np.sum(AroundVals) - 4 * A[ind]
            test2 = (test - U[ind])/U[ind]
            pass

        #use the values just make on the black square to make values for red squares
        for j in FilterRed:
            ind = tuple(j)
            AroundVals = [ A[tuple(j+d)] if ( ([0,0] <= j+d) & (j+d < dim) ).all() else A_old[ind] for d in Adjacent]
            Estimate = ( np.sum(AroundVals) - U[ind] )/4
            Direction = Estimate - A_old[ind]

            A[ind] = A_old[ind] + w_opt * Direction
            test = np.sum(AroundVals) - 4 * A[ind]
            pass

    return A

def Rev_Lap_test():
    testVal = np.zeros((25,25))
    testVal[0,10:15] = -25
    testVal[-1, 10:15] = -25
    fltr = [[0, 1, 0],
            [1,-4, 1],
            [0, 1, 0]]

    lap = testVal
    #lap = img.convolve(testVal, fltr, mode="nearest")

    final = np.NaN
    while True:
        final = MM.Inverse_Laplassian2d_SOR(lap, err=1e-2, init=final)
        final = final - final.min()

        lap2 = MM.Laplassian2d(final, 1, 'nearest')

        error = abs( (lap2-lap)/np.abs(lap).max() ) * 100
        if error.max() < 1:
            break

    Dist = np.sqrt(np.sum( (lap-lap2)**2 ))
    print("Distance: %5f | err mean: %5f | err max: %5f" % (Dist, np.mean(error), error.max()))
    plt.imshow(error)
    plt.show()


def Lap_test():
        
    fltr = np.array([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
    ])
    global testVal

    testVal = 200*np.sin(100*(testVal % np.pi))

    Lap = img.convolve(testVal, fltr, mode='constant')
    LinAlgAns = MM.Laplassian2d(Lap)

    errAlg = np.sqrt(np.sum( (LinAlgAns-testVal)**2 ))
    print(errAlg)

Rev_Lap_test()