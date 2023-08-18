import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib import animation as ani
import MathMethods as MM
import argparse

#parse args to see where and if to save data
# TODO add argument for interval
parser = argparse.ArgumentParser()
parser.add_argument('-S', '--Save', action='store_true', dest='Save')
parser.add_argument('-I', '--Input', action='store', default='sim data.json', dest='Input')
parser.add_argument('-O', '--Output', action='store', default='sim video.mp4', dest='Output')
parser.add_argument('-D', '--Display', action='store_true', dest='Display')
args = parser.parse_args()

# Import Data
with open(args.Input, "r") as infile:
    global RawData
    RawData = json.load(infile)

time =          np.array(RawData["Time"])
FVel =          np.array(RawData["Flow Velocity"])
GridCoords =    np.array(RawData["GridCoords"])
Pressure =      np.array(RawData["Pressure"])
#Pressure = MM.Grad(Pressure, 0, 1)

dims = GridCoords[0].shape




# prep template data
disp = FVel[0]
scaler = 0.8/abs(disp).max()
disp = scaler * disp

#Pressure = np.abs(Pressure)
Pscale = 1/abs(Pressure).max()
P = Pscale * Pressure[0]

#set up plot for display
fig = plt.figure()
if args.Save: fig.set(size_inches=(19.2, 10.8) )

ax = plt.axes(box_aspect=1)
PDisp = ax.imshow(P, extent=(GridCoords[0,0,0], GridCoords[0,-1,-1], GridCoords[1,0,0], GridCoords[1,-1,-1]), vmax=1.)
fig.colorbar(PDisp, ax=ax)
field = ax.quiver(GridCoords[0], GridCoords[1], 
                  disp[:,:,0], disp[:,:,1], 
                  angles='xy', scale_units='xy', scale=1)


#animation update fuction
def animate(t):

    #scale Flow Vel for display
    disp = FVel[t]
    disp = scaler * disp

    P = np.rot90(Pscale * Pressure[t]) 
    #div = 255 * div/div.max()

    #update values on plot
    field.set_UVC(disp[:,:,0], disp[:,:,1])
    PDisp.set_data(P)


    return PDisp, field

#run and show animation
anim = ani.FuncAnimation(fig, animate, time.size, interval=50, repeat=True)

if args.Save:
    anim.save(args.Output)


if (not args.Save) or args.Display:
    plt.show()



