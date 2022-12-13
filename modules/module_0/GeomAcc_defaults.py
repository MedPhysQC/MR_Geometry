
NUMBEROFINSTANCESPERSERIES = 25
TRANSVERSEORIENTATION = [1, 0, 0, 0, 1, 0]
GAUSSIAN_LAPLACE_SIGMA = 3
MARKER_THRESHOLD_AFTER_FILTERING = -1.5
CLUSTERSIZE = 150
ELLIPSOID_FACTOR = [1, 1, .1]
LIMIT_Z_SEPARATION_FROM_TABLEPOSITION = 25

#Labels of directions
AP = 0
LR = 1
CC = 2

LIMITFITDEGREES = 4 * 3.14159265358979 / 180 #radians -> 4 degrees
LIMITFITTRANS = 25 #mm

DIST_TO_ISOC_RIGID = 5  # Take into account the (5x5+1)^2 closest positions to isoc
CLUSTER_SEPARATION = 25

LIMITMAXDISTANCE = 2

BODY_CTR_POSITIONS = [
   #( AP,LR)
    (-125,-75),
    (-125,-50),
    (-125,-25),
    (-125, 0),
    (-125, 25),
    (-125, 50),
    (-125, 75),
    (-100,-125),
    (-100,-100),
    (-100, 100),
    (-100, 125),
    ( -75,-150),
    ( -75,-125),
    ( -75, 125),
    ( -75, 150),
    (-50, -175),
    (-50, -150),
    (-50, 150),
    (-50, 175),
    (-25, -175),
    (-25, 175),
    (  0, -175),
    (  0, 175),
    ( 25, -175),
    ( 25, 175),
    ( 50, -175),
    ( 50, -150),
    ( 50, 150),
    ( 50, 175),
    ( 75, -150),
    ( 75, -125),
    ( 75, 125),
    ( 75, 150)]


CENTER_POSITIONS = [
#   ( AP,LR)
    (-50,-25),
    (-50, 0),
    (-50, 25),
    (-25, -50),
    (-25, -25),
    (-25, 0),
    (-25, 25),
    (-25, 50),
    ( 0,-50),
    ( 0,-25),
    ( 0, 0),
    ( 0, 25),
    ( 0, 50),
    (25,-50),
    (25,-25),
    (25, 0),
    (25, 25),
    (25, 50),
    (50,-25),
    (50, 0),
    (50, 25)]


NEMA_PAIRS = [
    [(0, 225), (0, -225)],
    [(25, 225), (-25, -225)],
    [(25, -225), (-25, 225)],
    [(-200, 0), (75, 0)],
    [(-200, 25), (75, -25)],
    [(-200, -25), (75, 25)]]

AVLPHANTOM=(13.5,0.0,0,0)

markerPositions=[
#( AP, LR) [mm]
(-225.000000,-125.000000),
(-225.000000,-100.000000),
(-225.000000,-75.000000),
(-225.000000,-50.000000),
(-225.000000,-25.000000),
(-225.000000,0.000000),
(-225.000000,25.000000),
(-225.000000,50.000000),
(-225.000000,75.000000),
(-225.000000,100.000000),
(-225.000000,125.000000),
(-200.000000,-175.000000),
(-200.000000,-150.000000),
(-200.000000,-125.000000),
(-200.000000,-100.000000),
(-200.000000,-75.000000),
(-200.000000,-50.000000),
(-200.000000,-25.000000),
(-200.000000,0.000000),
(-200.000000,25.000000),
(-200.000000,50.000000),
(-200.000000,75.000000),
(-200.000000,100.000000),
(-200.000000,125.000000),
(-200.000000,150.000000),
(-200.000000,175.000000),
(-175.000000,-200.000000),
(-175.000000,-175.000000),
(-175.000000,-150.000000),
(-175.000000,-125.000000),
(-175.000000,-100.000000),
(-175.000000,-75.000000),
(-175.000000,-50.000000),
(-175.000000,-25.000000),
(-175.000000,0.000000),
(-175.000000,25.000000),
(-175.000000,50.000000),
(-175.000000,75.000000),
(-175.000000,100.000000),
(-175.000000,125.000000),
(-175.000000,150.000000),
(-175.000000,175.000000),
(-175.000000,200.000000),
(-150.000000,-250.000000),
(-150.000000,-225.000000),
(-150.000000,-200.000000),
(-150.000000,-175.000000),
(-150.000000,-150.000000),
(-150.000000,-125.000000),
(-150.000000,-100.000000),
(-150.000000,-75.000000),
(-150.000000,-50.000000),
(-150.000000,-25.000000),
(-150.000000,0.000000),
(-150.000000,25.000000),
(-150.000000,50.000000),
(-150.000000,75.000000),
(-150.000000,100.000000),
(-150.000000,125.000000),
(-150.000000,150.000000),
(-150.000000,175.000000),
(-150.000000,200.000000),
(-150.000000,225.000000),
(-150.000000,250.000000),
(-125.000000,-275.000000),
(-125.000000,-250.000000),
(-125.000000,-225.000000),
(-125.000000,-200.000000),
(-125.000000,-175.000000),
(-125.000000,-150.000000),
(-125.000000,-125.000000),
(-125.000000,-100.000000),
(-125.000000,-75.000000),
(-125.000000,-50.000000),
(-125.000000,-25.000000),
(-125.000000,0.000000),
(-125.000000,25.000000),
(-125.000000,50.000000),
(-125.000000,75.000000),
(-125.000000,100.000000),
(-125.000000,125.000000),
(-125.000000,150.000000),
(-125.000000,175.000000),
(-125.000000,200.000000),
(-125.000000,225.000000),
(-125.000000,250.000000),
(-125.000000,275.000000),
(-100.000000,-275.000000),
(-100.000000,-250.000000),
(-100.000000,-225.000000),
(-100.000000,-200.000000),
(-100.000000,-175.000000),
(-100.000000,-150.000000),
(-100.000000,-125.000000),
(-100.000000,-100.000000),
(-100.000000,-75.000000),
(-100.000000,-50.000000),
(-100.000000,-25.000000),
(-100.000000,0.000000),
(-100.000000,25.000000),
(-100.000000,50.000000),
(-100.000000,75.000000),
(-100.000000,100.000000),
(-100.000000,125.000000),
(-100.000000,150.000000),
(-100.000000,175.000000),
(-100.000000,200.000000),
(-100.000000,225.000000),
(-100.000000,250.000000),
(-100.000000,275.000000),
(-75.000000,-275.000000),
(-75.000000,-250.000000),
(-75.000000,-225.000000),
(-75.000000,-200.000000),
(-75.000000,-175.000000),
(-75.000000,-150.000000),
(-75.000000,-125.000000),
(-75.000000,-100.000000),
(-75.000000,-75.000000),
(-75.000000,-50.000000),
(-75.000000,-25.000000),
(-75.000000,0.000000),
(-75.000000,25.000000),
(-75.000000,50.000000),
(-75.000000,75.000000),
(-75.000000,100.000000),
(-75.000000,125.000000),
(-75.000000,150.000000),
(-75.000000,175.000000),
(-75.000000,200.000000),
(-75.000000,225.000000),
(-75.000000,250.000000),
(-75.000000,275.000000),
(-50.000000,-275.000000),
(-50.000000,-250.000000),
(-50.000000,-225.000000),
(-50.000000,-200.000000),
(-50.000000,-175.000000),
(-50.000000,-150.000000),
(-50.000000,-125.000000),
(-50.000000,-100.000000),
(-50.000000,-75.000000),
(-50.000000,-50.000000),
(-50.000000,-25.000000),
(-50.000000,0.000000),
(-50.000000,25.000000),
(-50.000000,50.000000),
(-50.000000,75.000000),
(-50.000000,100.000000),
(-50.000000,125.000000),
(-50.000000,150.000000),
(-50.000000,175.000000),
(-50.000000,200.000000),
(-50.000000,225.000000),
(-50.000000,250.000000),
(-50.000000,275.000000),
(-25.000000,-275.000000),
(-25.000000,-250.000000),
(-25.000000,-225.000000),
(-25.000000,-200.000000),
(-25.000000,-175.000000),
(-25.000000,-150.000000),
(-25.000000,-125.000000),
(-25.000000,-100.000000),
(-25.000000,-75.000000),
(-25.000000,-50.000000),
(-25.000000,-25.000000),
(-25.000000,0.000000),
(-25.000000,25.000000),
(-25.000000,50.000000),
(-25.000000,75.000000),
(-25.000000,100.000000),
(-25.000000,125.000000),
(-25.000000,150.000000),
(-25.000000,175.000000),
(-25.000000,200.000000),
(-25.000000,225.000000),
(-25.000000,250.000000),
(-25.000000,275.000000),
(0.000000,-275.000000),
(0.000000,-250.000000),
(0.000000,-225.000000),
(0.000000,-200.000000),
(0.000000,-175.000000),
(0.000000,-150.000000),
(0.000000,-125.000000),
(0.000000,-100.000000),
(0.000000,-75.000000),
(0.000000,-50.000000),
(0.000000,-25.000000),
(0.000000,0.000000),
(0.000000,25.000000),
(0.000000,50.000000),
(0.000000,75.000000),
(0.000000,100.000000),
(0.000000,125.000000),
(0.000000,150.000000),
(0.000000,175.000000),
(0.000000,200.000000),
(0.000000,225.000000),
(0.000000,250.000000),
(0.000000,275.000000),
(25.000000,-275.000000),
(25.000000,-250.000000),
(25.000000,-225.000000),
(25.000000,-200.000000),
(25.000000,-175.000000),
(25.000000,-150.000000),
(25.000000,-125.000000),
(25.000000,-100.000000),
(25.000000,-75.000000),
(25.000000,-50.000000),
(25.000000,-25.000000),
(25.000000,0.000000),
(25.000000,25.000000),
(25.000000,50.000000),
(25.000000,75.000000),
(25.000000,100.000000),
(25.000000,125.000000),
(25.000000,150.000000),
(25.000000,175.000000),
(25.000000,200.000000),
(25.000000,225.000000),
(25.000000,250.000000),
(25.000000,275.000000),
(50.000000,-275.000000),
(50.000000,-250.000000),
(50.000000,-225.000000),
(50.000000,-200.000000),
(50.000000,-175.000000),
(50.000000,-150.000000),
(50.000000,-125.000000),
(50.000000,-100.000000),
(50.000000,-75.000000),
(50.000000,-50.000000),
(50.000000,-25.000000),
(50.000000,0.000000),
(50.000000,25.000000),
(50.000000,50.000000),
(50.000000,75.000000),
(50.000000,100.000000),
(50.000000,125.000000),
(50.000000,150.000000),
(50.000000,175.000000),
(50.000000,200.000000),
(50.000000,225.000000),
(50.000000,250.000000),
(50.000000,275.000000),
(75.000000,-275.000000),
(75.000000,-250.000000),
(75.000000,-225.000000),
(75.000000,-200.000000),
(75.000000,-175.000000),
(75.000000,-150.000000),
(75.000000,-125.000000),
(75.000000,-100.000000),
(75.000000,-75.000000),
(75.000000,-50.000000),
(75.000000,-25.000000),
(75.000000,0.000000),
(75.000000,25.000000),
(75.000000,50.000000),
(75.000000,75.000000),
(75.000000,100.000000),
(75.000000,125.000000),
(75.000000,150.000000),
(75.000000,175.000000),
(75.000000,200.000000),
(75.000000,225.000000),
(75.000000,250.000000),
(75.000000,275.000000)]