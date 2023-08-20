# code by Sebastian Cypert
# Resouces used:
# https://en.wikipedia.org/wiki/Discrete_Poisson_equation
# https://people.eecs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html#link_4
# https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations
# https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_existence_and_smoothness

import numpy as np
import json
from scipy import integrate as integ
from MathMethods import *
import argparse

# set Constants for simulation
X = 0
Y = 1
Z = 2


# fluid properties
density = 1
grav = 9.8 # currently doesn't do anything. no current external force.
kenVisc = 1


# simulation space properties
h = 0.25
SpaceSize = 100


#hangle args
parser = argparse.ArgumentParser()
parser.add_argument('-O', '--Output', action='store', dest='Output', default='Save Sim Data/Latest Sim Data.json')
parser.add_argument('-E', '--EndTime', action='store', dest='EndTime', default=1)
parser.add_argument('-S', '--Samples', action='store', dest='Samples', default=50)

args = parser.parse_args()

#poisson pressure equation
Pressure_Est = np.NaN
def Pressure_from_Velocity(vel):
    global Pressure_Est

    partialy = Grad(vel, Y, h).transpose(2,0,1)
    partialx = Grad(vel, X, h).transpose(2,0,1)
    YTerm = partialy[Y]**2 * -np.sign(partialy[Y]) # added signs here to conserve sign
    MTerm = 2 * partialy[X] * partialx[Y]          # should look at math to understand better
    XTerm = partialx[X]**2 * -np.sign(partialx[X])
    Laplacian = -density * (YTerm + MTerm + XTerm)


    ## TODO make an initial guess that's better then starting from 0s


    Pressure_Est = Inverse_Laplassian2d_SOR(Laplacian, err=1e-3, max_iter=200, init=Pressure_Est, dx=h)

    Pressure_Est = Pressure_Est - np.mean(Pressure_Est)
    return Pressure_Est


#navier stokes equation
def Calculate_Acceleration(velocity, pressure):

    #padding pressure to give edge clamp boundry condition
    pressure_padded = np.pad(pressure, 1, mode="edge")
    dPdy = Grad(pressure_padded, Y, dx=h)[1:-1, 1:-1]
    dPdx = Grad(pressure_padded, X, dx=h)[1:-1, 1:-1]


    MomentumPre =  -1/density * np.array([ dPdx, dPdy ]).transpose(1,2,0)
    MomentumDif = kenVisc * (Grad2(velocity, X, h) + Grad2(velocity, Y, h))
    MomentumExt = 0


    RelativeTerm = -sum([ velocity[:,:,i][:,:,None] * Grad(velocity, i, h) for i in range(2)])

    Acceleration = MomentumDif + MomentumPre + RelativeTerm
    return Acceleration

#for edge conditions
#the pressure edge condition should copy nearest as the tank is exserting pressure on it 
#the flow velocity should have [0,0] at edge since the tank always stops the flow and isn't flowing.

def Diff_Equation(t, u, shape):
    u = u.reshape(shape, order='C')

    pressure = Pressure_from_Velocity(u)
    answer = Calculate_Acceleration(u, pressure)


    print( "Sim Time: {0:d}s {1:03d}ms".format( int(t) , int( (t%1)*1000 ) ) )

    return answer.flatten()


def main():

    # set up initial conditions of simulation
    init = np.zeros((SpaceSize,SpaceSize,2))
    space = init.shape
    init[40:60,:,Y] = 5

    # set up how long the simulation will run and how much data will be saved
    endTime = int(args.EndTime)
    samples = int(args.Samples)


    # flatten the initial and run the simulation
    Flat_Init = init.flatten(order='C')
    a = integ.solve_ivp(Diff_Equation, (0, endTime), 
                        Flat_Init, args=(init.shape,), 
                        t_eval=np.linspace(0,endTime,samples))

    print("Sim Finished, Saving...")


    # reshape solved data, and set up space.
    sol = np.array([a.y[:,i].reshape(space) for i in range(a.t.size)])
    XGrid, YGrid = np.mgrid[0:h*space[0]:h, 0:h*space[1]:h].tolist()
    Coords = [XGrid, YGrid]


    # find pressure for all simulation points
    # TODO break this out into a for loop and print progress
    Pressure = [ Pressure_from_Velocity(val).tolist()
                    for val in sol ]

    # set up data and export to json file
    exportData = {
        "Flow Velocity" : sol.tolist(),
        "Time" : a.t.tolist(),
        "GridCoords": Coords,
        "Pressure": Pressure
    }
    JsonText = json.dumps(exportData, indent=4)

    with open(args.Output, "w") as outfile:
        outfile.write(JsonText)

main()