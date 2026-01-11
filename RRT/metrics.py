import math
import numpy as np
#Function to determine the distance between two given points
#Inputs:
#   - coord1: the first coordinate
#   - coord2: the second coordinate
#   - system: the coordinate system used for the coordinates
#
#Outputs:
#   - distance: the distance between the two points
def calculateDistance(coord1, coord2, system = "cartesian"):    
    #Check if the coordinates have equal and valid dimensions. Raise an error if not
    assert len(coord1) == len(coord2), "Error: the two coordinate arrays are not the same length"
        
    if (len(coord1) <= 0 or len(coord2) <= 0 or len(coord1) >= 4 or len(coord2) >= 4):
        raise ValueError("Invalid number of dimensions for coordinates")
        
    #Change the input coordinates to a 3D vector (tuple) for easier use
    point1 = tuple(coord1) + (0,) * (3 - len(coord1))  #Initialize with zeros
    point2 = tuple(coord2) + (0,) * (3 - len(coord2))  #Initialize with zeros
    
    #Check if the coordinate system is either in cartesian, polar, cylindrical or spherical, 
    #and raise an error if not.         
    if system.lower() == "cartesian":
        #Calculate the distance in Cartesian coordinates
        distance = abs(np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2))
        
    elif (system.lower() == "polar" or system.lower() == "cylindrical"):
        #Calculate the distance in polar or cylindrical coordinates
        distance = abs(np.sqrt(point1[0]**2 + point2[0]**2 - 2*point1[0]*point2[0]*np.cos(point2[1] - point1[1]) + (point2[2] - point1[2])**2))
        
    elif system.lower() == "spherical":
        #Calculate the distance in spherical coordinates
        distance = abs(np.sqrt(point1[0]**2 + point2[0]**2 - 2*point1[0]*point2[0]*np.cos(point2[1] - point1[1])*np.cos(point2[2] - point1[2])))
        
    else: 
        #If the system is not cartesian, polar, cylindrical or spherical, raise an error
        raise ValueError("Invalid input for system. Please choose either 'cartesian', 'polar', " +  
                         "'cylindrical' or 'spherical'")
    
    #Return the final distance
    return distance


#Function to determine the curvature of a given link or trajectory
#Inputs:
#   - linearVelocities: an array of the linear velocities of the robot
#   - angularVelocities: an array of the angular velocities of the robot
#   
#Outputs:
#   - curvatureFactor: the curvature factor of the robot
def curvature(linearVelocities, angularVelocities):
    #Check if the two velocity arrays are the same length, and raise an error if not
    assert len(linearVelocities) == len(angularVelocities), "Error: the two velocity arrays are not the same length"
    
    #Check the dimension of the input. If 1D, continue to calculate the curvature factor. If 2D,
    #transform to 1D with the norm of the velocity per step.
    dimensionLV = len(np.array(linearVelocities).shape)
    if dimensionLV == 1:
        linVel = linearVelocities
    elif dimensionLV == 2:
        linVel = [np.linalg.norm(linearVelocities[i]) for i in range(len(linearVelocities))]
    
    #Check the dimension of the input. If 1D, continue to calculate the curvature factor. If 2D,
    #transform to 1D with the norm of the velocity per step.
    dimensionAV = len(np.array(angularVelocities).shape)
    if dimensionAV == 1:
        angVel = angularVelocities
    elif dimensionAV == 2:
        angVel = [np.linalg.norm(angularVelocities[i]) for i in range(len(angularVelocities))]
        
    #Calculate the curvature factor for each step
    curvatureFactor = [0] * len(linVel)
    for i in range(len(linVel)):
        #Check if the linear velocity is zero. If so, set the curvature factor to -1
        try:
            curvatureFactor[i] = abs(angVel[i] / linVel[i])
        except ZeroDivisionError:
            curvatureFactor[i] = -1
        
    #Return the final curvature factor
    return curvatureFactor

#Function to determine the completion time of the robot movement
#Inputs: 
#   - startTime: the start time of the simulation
#   - endTime: the end time of the simulation
#   - timeSteps: the time steps of the simulation
#
#Outputs:
#   - completionTime: the total time the robot has been in motion
def completionTime(startTime = 0, endTime = 0, timeSteps = 0):
    #Check for a valid input combination for the timeSteps
    if (timeSteps != 0 and startTime == 0 and endTime == 0):
        #Check that all the values inside the array are positive. If not, raise an error
        if any(x < 0 for x in timeSteps):
            raise ValueError("Negative or zero value for time found." +
                             " Check if your input is correct.")
            
        #If dt is a input (assume 1D array) then sum the total time in that array
        completionTime = sum(timeSteps)
    
    #Check for a valid input combination for startTime and endTime
    elif (endTime != 0 and timeSteps == 0):
        #Check that the start and end time values are larger than zero (valid)
        if (startTime < 0 or endTime < 0):
            raise ValueError("Negative value for input time found. Check if your input is correct.")
        
        #If start and end time are given, calculate the difference
        elif (startTime > endTime):
            raise ValueError("Error: start time is later then the end time. Check if your input is correct.")
        
        else:
            completionTime = endTime - startTime
    else:
        #If none of the above are given, raise an error due to incorrect input
        raise ValueError("No proper input given for dt, start_time and/or end_time")
    
    #Return the completion time if no errors were raised
    return completionTime

#Function to determine the total path length of a given array of stored points.
#Inputs:
#   - positions: an array of the positions of the robot
#   - system: the coordinate system used for the positions
#
#Outputs:
#   - pathlength: the total path length of the robot
def pathLength(positions, system = "cartesian"):
    #Initialize path length
    pathlength = 0
    
    #Check if the positions array is valid, and raise an error if not. One point is not a path, 
    assert len(positions) > 1, "Error: the positions array is too short, no path can be made from this"
    
    #Calculate the path length by calculating the distance over each step
    for i in range(len(positions) - 1):
        stepLength = calculateDistance(positions[i], positions[i + 1], system)
        pathlength += stepLength  #Add step length to total path length
        
    #Return the final path length
    return pathlength 

#Function to determine the curvature change of a given link
#Inputs:
#   - linearVelocities: an array of the linear velocities of the robot
#   - angularVelocities: an array of the angular velocities of the robot
#   - timeSteps: an array of the time steps
#   - positions: an array of the positions of the robot
#   - system: the coordinate system used for the positions
#
#Outputs:
#   - fcc: the curvature change factor of the robot
def curvatureChange(linearVelocities, angularVelocities, timeSteps, positions, system = "cartesian"):
    #Check if the two velocity arrays are the same length, and raise an error if not
    assert len(linearVelocities) == len(angularVelocities), "Error: the two velocity arrays are not the same length"
    
    #Check if the dt input is valid
    if any(x < 0 for x in timeSteps):
        raise ValueError("Negative or zero value for time found. Check if your input is correct.")
    
    #Determine the completion time
    completion_time = completionTime(startTime = timeSteps[0], endTime = timeSteps[-1])
    
    #Determine the path length
    path_length = pathLength(positions, system)
    
    #Initiate an array for the curvature factor
    curvatureFactor = curvature(linearVelocities, angularVelocities) 
    fcc = 0

    #Filter out zero-values created by the zeroDivisionError, neglecting them in the calculation
    zeroIndex = []
    for i in range(len(curvatureFactor)):
        if curvatureFactor[i] == -1:
            zeroIndex.append(i)
      
    for i in range(len(zeroIndex)):
        curvatureFactor.pop(zeroIndex[-(i + 1)])
        timeSteps.pop(zeroIndex[-(i + 1)])

    #CInitiate and calculate the derivative of the curvature change factor
    curvatureFactorPrime = np.zeros(len(curvatureFactor) - 1)
    for i in range(len(curvatureFactor) - 1):
        curvatureFactorPrime[i] = abs((curvatureFactor[i + 1] - curvatureFactor[i]) / (timeSteps[i + 1] - timeSteps[i]))

    #Calculate the sum for the curvature change factor
    for i in range(len(curvatureFactorPrime) - 1):
        fcc = fcc + (curvatureFactorPrime[i] + curvatureFactorPrime[i + 1])

    #Calculate the final curvature change factor
    fcc = fcc * (completion_time / (2 * len(linearVelocities) * path_length))
    return fcc