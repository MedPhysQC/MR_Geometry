#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""

Dependencies:

    PYTHON
    - pydicom / dicom
    - numpy
    - matplotlib


Version control:
    20170303: first version (WAD QC 1,0)
    20190128: second version (WAD QC 2.0)


The initial version was created by Erik van der Bijl.
The initial version was adapted by Stijn van de Schoot at the Amsterdam UMC, location AMC in order to use the code for WAD QC 2.0 software.

"""

__name__ = 'GeomAcc_lib.py'
__version__ = '20190128'
__author__ = 'Stijn van de Schoot (s.vandeschoot@amc.uva.nl)'


import matplotlib
matplotlib.use('Agg')

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import colors
from scipy import ndimage
from scipy import optimize

# Import DICOM
try:
    import pydicom as dicom
except ImportError:
    import dicom

# Import GeomAcc_defaults
try:
    import GeomAcc_defaults as GeomAccDefaults
except ImportError:
    print('[{0}] :: IMPORT ERROR: Could not import GeomAcc_defaults module.'.format(os.path.basename(__file__)))







class TablePostionData():

    def __init__(self, phantomType):

        #Constants/Config
        self.NUMBEROFINSTANCESPERSERIES = GeomAccDefaults.NUMBEROFINSTANCESPERSERIES
        self.TRANSVERSE_ORIENTATION = GeomAccDefaults.TRANSVERSEORIENTATION
        self.GAUSSIAN_LAPLACE_SIGMA = GeomAccDefaults.GAUSSIAN_LAPLACE_SIGMA
        self.MARKER_THRESHOLD_AFTER_FILTERING = GeomAccDefaults.MARKER_THRESHOLD_AFTER_FILTERING
        self.CLUSTERSIZE = GeomAccDefaults.CLUSTERSIZE
        self.ELLIPSOID_FACTOR = GeomAccDefaults.ELLIPSOID_FACTOR
        self.LIMIT_Z_SEPARATION_FROM_TABLEPOSITION = GeomAccDefaults.LIMIT_Z_SEPARATION_FROM_TABLEPOSITION

        self.degLimit = GeomAccDefaults.LIMITFITDEGREES
        self.transLimit = GeomAccDefaults.LIMITFITTRANS

        self.LR = GeomAccDefaults.LR
        self.AP = GeomAccDefaults.AP
        self.CC = GeomAccDefaults.CC

        self.phantomType = phantomType

        #Results
        self.headerData = {}
        self.tablePosition = None
        self.expectedMarkerPositions = None
        self.detectedMarkerPositions = None

        self.correctedMarkerPositions = None
        self.closestExpectedMarkerIndices = None
        self.differencesCorrectedExpected = None

    def setTablePosition(self, tablePosition):
        self.tablePosition = tablePosition

    def loadImageDataAndDetectClusters(self, dcmSeries):

        dcmSeries = self.selectInstances(dcmSeries)
        success = False

        nInstances = len(dcmSeries)

        if nInstances == 0:
            print("[{0}] :: TablePositionData - loadImageDataAndDetectClusters -- WARNING: No DICOM instances in series, skipping this series!".format(os.path.basename(__file__)))
        elif not nInstances == self.NUMBEROFINSTANCESPERSERIES:
            print("[{0}] :: TablePositionData - loadImageDataAndDetectClusters -- WARNING: Missing a number of instances --> Expected: {1}; Found: {2}!".format(
                    os.path.basename(__file__), str(self.NUMBEROFINSTANCESPERSERIES), str(nInstances)))
        else:
            self.headerData = self.readHeaderData(dcmSeries)
            self.setTablePosition(self.headerData['table_position'])
            self.setExpectedMarkerPositions(self.tablePosition)

            print("[{0}] :: TablePositionData - loadImageDataAndDetectClusters -- INFO: Loading images in series at table position {1}!".format(os.path.basename(__file__), str(self.tablePosition)))
            imageData, pixelCoordinates = self._readPixelData(dcmSeries)
            print("[{0}] :: TablePositionData - loadImageDataAndDetectClusters -- INFO: Detecting marker positions!".format(os.path.basename(__file__)))
            self.detectedMarkerPositions = self.detectMarkerPositions(imageData=imageData, pixelCoordinates=pixelCoordinates)
            success = True

        return success

    def selectInstances(self, dcmSeries):
        selectedInstances = []
        for instance in dcmSeries:
            try:
                acqTime = instance.AcquisitionTime
                selectedInstances.append(instance)
            except:
                print("[{0}] :: TablePositionData - selectInstances -- WARNING: Ignore this instance.".format(os.path.basename(__file__)))
        return selectedInstances

    def readHeaderData(self, dcmSeries):
        headerData = {}
        dcmInstance = dcmSeries[0]
        headerData.update({'table_position': self._getTablePosFromSeriesDescr(dcmInstance.SeriesDescription)})
        acquisitionDate = dcmInstance.AcquisitionDate
        acquisitionTime = dcmInstance.AcquisitionTime
        PerformedStationAETitle = dcmInstance.PerformedStationAETitle

        # print "Other Header:",dcmInstance[0x07a5,0x1069]
        headerData.update({'acquisition_date': acquisitionDate})
        headerData.update({'acquisition_time': acquisitionTime})
        headerData.update({'PerformedStationAETitle': PerformedStationAETitle})
        return headerData

    def _getTablePosFromSeriesDescr(self, seriesDescription):
        zpos = seriesDescription.split('_')[2]
        zpos = float(zpos.replace('m', '-'))
        return zpos

    def setExpectedMarkerPositions(self, tablePosition):
        markerLocations2D = np.array(GeomAccDefaults.markerPositions)

        # adjust for NKI phantom type
        if self.phantomType == 'NKIPhantom':
            markerLocations2D += np.array(GeomAccDefaults.AVLPHANTOM)
        self.expectedMarkerPositions = np.array(np.insert(markerLocations2D, 2, tablePosition, axis=1))

    def _readPixelData(self, dcmSeries):
        imageData = []
        image_positions = []
        for dcmInstance in dcmSeries:
            imageData.append(dcmInstance.pixel_array.astype('float32'))
            image_positions.append(dcmInstance.ImagePositionPatient)
            pixelSpacing = dcmInstance.PixelSpacing
        imageData = np.array(imageData)
        image_positions = np.array(image_positions, dtype=np.float32)
        imageData, image_positions = self._sortImageData(imageData, image_positions)
        pixelCoordinates = self._calculatePixelCoordinates(image_positions, imageData.shape, pixelSpacing)
        return imageData, pixelCoordinates

    def _sortImageData(self, imageData, imagePositions):
            correctOrder = np.argsort([z for x, y, z in imagePositions])
            return imageData[correctOrder], imagePositions[correctOrder]

    def _calculatePixelCoordinates(self, image_positions, pixelDataShape, pixelSpacing):
        pixelCoordinates_X = image_positions[0][1] + np.arange(pixelDataShape[1]) * pixelSpacing[1]
        pixelCoordinates_Y = image_positions[0][0] + np.arange(pixelDataShape[2]) * pixelSpacing[0]
        pixelCoordinates = [pixelCoordinates_X, pixelCoordinates_Y, image_positions[:, 2]]
        return pixelCoordinates

    def detectMarkerPositions(self, imageData, pixelCoordinates):
        highContrastVoxels = self._getHighContrastVoxelsFromImageData(imageData)
        highContrastPoints = self._convertIndicesToCoords(highContrastVoxels, pixelCoordinates)
        clusters = self._createClusters(highContrastPoints, self.CLUSTERSIZE)
        detectedMarkerPositions = self._removeClustersFarFromExpectedTablePosition(clusters, self.tablePosition, self.LIMIT_Z_SEPARATION_FROM_TABLEPOSITION)
        return detectedMarkerPositions

    def _getHighContrastVoxelsFromImageData(self, imageData):
        print("[{0}] :: TablePositionData - _getHighContrastVoxelsFromImageData -- INFO: Filtering dataset.".format(os.path.basename(__file__)))
        filteredImage = ndimage.gaussian_laplace(imageData, self.GAUSSIAN_LAPLACE_SIGMA)
        idx = np.argwhere(filteredImage < self.MARKER_THRESHOLD_AFTER_FILTERING).T
        return idx

    def _convertIndicesToCoords(self, indexList, pixelCoordinates):
        xCoord = pixelCoordinates[0][indexList[1]]
        yCoord = pixelCoordinates[1][indexList[2]]
        zCoord = pixelCoordinates[2][indexList[0]]
        return np.array(list(zip(xCoord, yCoord, zCoord)))

    def _createClusters(self, points, size):
        print("[{0}] :: TablePositionData - _createClusters -- INFO: Finding clusters of high intensity voxels.".format(os.path.basename(__file__)))
        clusters = []
        while len(points) > 0:
            curPoint = points[0]
            squaredDistanceToCurrentPoint = self._ellipsoidDistance(points, curPoint)

            pointsCloseToCurrent = squaredDistanceToCurrentPoint < size
            clusters.append(np.mean(points[pointsCloseToCurrent], axis=0))
            points = points[np.logical_not(pointsCloseToCurrent)]
        return np.array(clusters)

    def _ellipsoidDistance(self, points, centerOfCylinder):
        distances = (points - centerOfCylinder) * self.ELLIPSOID_FACTOR
        return np.sum(np.power(np.array(distances), 2), axis=1)

    def _removeClustersFarFromExpectedTablePosition(self, clusters, tablePosition, limit):
        print("[{0}] :: TablePositionData - _removeClustersFarFromExpectedTablePosition -- INFO: Removing detected markers far from expected C-C position.".format(os.path.basename(__file__)))

        try:
            return clusters[abs(clusters[:, self.CC] - tablePosition) < limit]
        except:
            print("[{0}] :: TablePositionData - _removeClustersFarFromExpectedTablePosition -- WARNING: Empty cluster since no points are detected (probably due to no detected high contrast voxels).".format(os.path.basename(__file__)))
            return clusters

    def findRigidTransformation(self, detectedMarkerPositions, expectedMarkerPositions):
        print("[{0}] :: TablePositionData - findRigidTransformation -- INFO: Determining setup translation and rotation for table position 0.".format(os.path.basename(__file__)))
        init_CC = self.tablePosition - np.mean(detectedMarkerPositions, axis=0)[self.CC]
        optimization_initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        optimization_initial_guess[self.CC] = init_CC

        optimization_bounds = [(-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit)]

        def penaltyFunction(transRot):
            trPos = self.rigidTransform(detectedMarkerPositions, transRot[0:3], [transRot[3], transRot[4], transRot[5]])
            differences = self.getdifferences(trPos, expectedMarkerPositions)
            penalty = np.sum(np.sum(np.power(differences, 2)))
            return penalty

        opt_result = optimize.minimize(fun=penaltyFunction,
                                       x0=optimization_initial_guess,
                                       bounds=optimization_bounds,
                                       tol=.00001)


        return opt_result.x

    def rigidRotation(self, markerPositions, eulerAngles):
        s0 = np.sin(eulerAngles[0])
        c0 = np.cos(eulerAngles[0])
        s1 = np.sin(eulerAngles[1])
        c1 = np.cos(eulerAngles[1])
        s2 = np.sin(eulerAngles[2])
        c2 = np.cos(eulerAngles[2])

        m00 = c1*c2
        m01 = c0*s2+s0*s1*c2
        m02 = s0*s2-c0*s1*c2
        m10 = -c1*s2
        m11 = c0*c2-s0*s1*s2
        m12 = s0*c2+c0*s1*s2
        m20 = s1
        m21 = -s0*c1
        m22 = c0*c1

        rotationMatrix = np.array([
                         [m00, m01, m02],
                         [m10, m11, m12],
                         [m20, m21, m22]])

        return np.dot(markerPositions, rotationMatrix)

    def rigidTranslation(self, markerPositions, Translation):
        return markerPositions+Translation

    def rigidTransform(self, positions, translation, eulerAngles):

        local_positions = self.rigidTranslation(positions, [0.0, 0.0, -self.tablePosition])
        rotated_local_positions = self.rigidRotation(local_positions, eulerAngles)
        rotated_positions = self.rigidTranslation(rotated_local_positions, [0.0, 0.0, + self.tablePosition])
        transformed_positions = self.rigidTranslation(rotated_positions, translation)
        return transformed_positions

    def setCorrectedMarkerPositions(self, transformation):
        self.correctedMarkerPositions = self.rigidTransform(self.detectedMarkerPositions, transformation[0:3], [0.0, transformation[4], transformation[5]])
        self.closestExpectedMarkerIndices = self.closestLocations(self.correctedMarkerPositions, self.expectedMarkerPositions)
        self.differencesCorrectedExpected = self.getdifferences(self.correctedMarkerPositions, self.expectedMarkerPositions)

    def closestLocations(self, detectedMarkerPositions, expectedMarkerPositions):
        distances = np.sum(np.power((detectedMarkerPositions - expectedMarkerPositions[:, np.newaxis]), 2), axis=2)
        return np.argmin(distances, axis=0)

    def getdifferences(self, markerPositions, expectedMarkerPositions):
        return markerPositions - expectedMarkerPositions[self.closestLocations(markerPositions, expectedMarkerPositions)]

    def _findMarkerIndex(self, point):
        xPosMarkers, yPosMarkers, _ = self.expectedMarkerPositions.T
        xMatch = point[0] == xPosMarkers
        yMatch = point[1] == yPosMarkers
        return np.argwhere(np.logical_and(xMatch, yMatch))[0]

    def calculateStatisticsForExpectedMarkerPositions(self, MarkerPositions):
        indices = np.array([self._findMarkerIndex(pos) for pos in MarkerPositions]).flatten()
        indices_of_differences = []
        No_Missing_Points = 0
        for index in indices:
            try:
                indices_of_differences.append(np.argwhere(index == self.closestExpectedMarkerIndices)[0, 0])
            except:
                No_Missing_Points += 1
        differences_defined_by_indices = self.differencesCorrectedExpected[indices_of_differences]
        mean = np.mean(differences_defined_by_indices, axis=0)
        rms = np.std(differences_defined_by_indices, axis=0)
        return mean, rms, No_Missing_Points

    def calculateMaxDistanceWithinLimit(self, limit_in_mm):

        differences = self.differencesCorrectedExpected
        pos2D = np.delete(self.correctedMarkerPositions, 2, axis=1)

        correctedDistancesToIsoc = np.sqrt(np.sum(np.power(pos2D, 2), axis=1))

        max_dist = 0.0
        for dist in list(np.linspace(2, 300, num=251)):
            correctedMarkerIndices_within_dist = correctedDistancesToIsoc < dist
            deviations = np.sqrt(np.sum(np.power(differences[correctedMarkerIndices_within_dist], 2), axis=1))
            if np.max(deviations) <= limit_in_mm:
                max_dist = dist

        return max_dist


class GeomAcc():

    def __init__(self, phantomType):

        #Define default constants
        self.LR = GeomAccDefaults.LR
        self.AP = GeomAccDefaults.AP
        self.CC = GeomAccDefaults.CC

        #Properties of the study
        self.phantomType = phantomType
        self.studyDate = None
        self.studyTime = None
        self.studyScanner = None

        #results for this study
        self.rigid_transformation_setup = [0, 0, 0, 0, 0, 0]
        self.measurementsPerTablePos = {}
        self.measurementTablePositions = []

        #Phantom type dependent default values
        self.NEMA_PAIRS = np.array(GeomAccDefaults.NEMA_PAIRS)
        self.CENTER_POSITIONS = np.array(GeomAccDefaults.CENTER_POSITIONS)
        self.BODY_CTR_POSITIONS = np.array(GeomAccDefaults.BODY_CTR_POSITIONS)
        if self.phantomType == 'NKIPhantom':
            self.adjustForPhantomType()


    def adjustForPhantomType(self):
        print("[{0}] :: GeomAcc - adjustForPhantomType -- INFO: Adjusting expected positions to AvL phantom type.".format(os.path.basename(__file__)))
        newNEMAPairs = []
        for pt1, pt2 in self.NEMA_PAIRS:
            newNEMAPairs.append([pt1+GeomAccDefaults.AVLPHANTOM, pt2+GeomAccDefaults.AVLPHANTOM])
        self.NEMA_PAIRS = np.array(newNEMAPairs)

        self.BODY_CTR_POSITIONS = self.BODY_CTR_POSITIONS + GeomAccDefaults.AVLPHANTOM

        self.CENTER_POSITIONS = self.CENTER_POSITIONS + GeomAccDefaults.AVLPHANTOM


    def loadImagesAndDetectClusterLocations(self, data):
        print("[{0}] :: GeomAcc - loadImagesAndDetectClusterLocations -- INFO: Start loading DICOM series.".format(os.path.basename(__file__)))

        #if len(data.getAllSeries()) < int(numberOfSeries):
            #print("[{0}] :: GeomAcc - loadImagesAndDetectClusterLocations -- ERROR: Could not find all series. Expected number of series is {1} and detected number of series is {2}.".format(os.path.basename(__file__), iExpectedNumberOfSeries, len(data.getAllSeries())))
            #print("[{0}] :: GeomAcc - loadImagesAndDetectClusterLocations -- ERROR: Abort analysis due to mismatching number of data series.".format(os.path.basename(__file__)))

        self.loadSeries(data.getAllSeries())


    def loadSeries(self,seriesList):
        selectedSeries = self.selectRelevantSeriesOnly(seriesList)

        for series in selectedSeries:
            print("[{0}] :: GeomAcc - loadSeries -- INFO: Loading serie with SeriesNumber {1}.".format(os.path.basename(__file__), series[0].SeriesNumber))
            currentMeasurement = TablePostionData(phantomType=self.phantomType)
            if currentMeasurement.loadImageDataAndDetectClusters(series):
                self.addMeasurement(currentMeasurement)
                self.getStudyProperties(currentMeasurement)
            else:
                print("[{0}] :: GeomAcc - loadSeries -- WARNING: Problem with this measurement, skipping this one.".format(os.path.basename(__file__)))

    def getStudyProperties(self, measurement):

        print("[{0}] :: GeomAcc - getStudyProperties -- INFO: Getting study properties...".format(os.path.basename(__file__)))
        self.studyDate = measurement.headerData['acquisition_date']
        self.studyTime = measurement.headerData['acquisition_time']
        self.studyScanner = measurement.headerData['PerformedStationAETitle']

    def selectRelevantSeriesOnly(self, seriesList):
        """
        This function is inserted to handle all the extra series
        that the Philips Batch job adds to the acquisitions. The
        Series which contain the acquisitions are numbered 'x01'
        with x in [1-7].

        This selection is in principle not necessary since the
        WAD server can be configured to only send the correct
        Series in the study to this processor.
        :param seriesList:
        :return: selected series from list
        """
        selectedSeries = []
        for series in seriesList:
            instance = series[0]
            seriesNumber = str(instance.SeriesNumber)
            if seriesNumber.endswith('1'):
                selectedSeries.append(series)
        return selectedSeries

    def addMeasurement(self, measurement):
        if measurement.tablePosition not in self.getTablePositions():
            self.measurementsPerTablePos.update({measurement.tablePosition: measurement})
            print("[{0}] :: GeomAcc - addMeasurement -- INFO: Adding measurement for specific table position...".format(os.path.basename(__file__)))
        else:
            print("[{0}] :: GeomAcc - addMeasurement -- WARNING: Double table position in study, skipping this one.".format(os.path.basename(__file__)))

    def getTablePositions(self):
        return sorted(self.measurementsPerTablePos.keys())

    def getMeasurement(self, tablePosition):
        if tablePosition in self.getTablePositions():
            return self.measurementsPerTablePos[tablePosition]
        else:
            print("[{0}] :: GeomAcc - getMeasurement -- ERROR: This table position does not exist.".format(os.path.basename(__file__)))

    def correctDetectedClusterPositionsForSetup(self):
        """
        This function adds corrected clusterpositions to the measurements based on the
        calculated setup rotation and translation in the measurement at tableposition 0
        :return:
        """
        print("[{0}] :: GeomAcc - correctDetectedClusterPositionsForSetup -- INFO: Correcting for phantom setup.".format(os.path.basename(__file__)))
        central_measurement = self.getMeasurement(tablePosition=0)
        self.rigid_transformation_setup = central_measurement.findRigidTransformation(central_measurement.detectedMarkerPositions, central_measurement.expectedMarkerPositions)
        for tablePosition in self.getTablePositions():
            self.getMeasurement(tablePosition=tablePosition).setCorrectedMarkerPositions(self.rigid_transformation_setup)

    def save_detected_clusters_to_wad(self, results):
        detectedPositions=[]
        for tablePosition in self.getTablePositions():
            correctedClusterPositions = list(self.getMeasurement(tablePosition).correctedMarkerPositions)

            detectedPositions.extend(correctedClusterPositions)
        np.savetxt("DetectedClusterPositions.txt", detectedPositions)
        results.addObject("Detected Cluster Positions", 'DetectedClusterPositions.txt')

    def save_numeric_to_wad(self, results):
        print("[{0}] :: GeomAcc - save_numeric_to-wad -- INFO: Calculating results...".format(os.path.basename(__file__)))

        self._addDetectedExpectedMarkerRatioToResults(results)
        self._addSetupCorrectionToResults(results)
        self._addNEMAToResults(results)
        self._addLinearityValuesToResults(results)
        self._addRegionStatisticsToResults(results)


    def _addDetectedExpectedMarkerRatioToResults(self, results):
        print("[{0}] :: GeomAcc - _addDetectedExpectedmarkerRatioToResults -- INFO: Adding number of detected points to WAD results.".format(os.path.basename(__file__)))

        detectedMarkers = 0
        expectedMarkers = 0

        for tablePosition in self.getTablePositions():
            detectedMarkers += len(self.getMeasurement(tablePosition).detectedMarkerPositions)
            expectedMarkers += len(self.getMeasurement(tablePosition).expectedMarkerPositions)

        results.addFloat("Gedetecteerde markers", detectedMarkers)


    def _addSetupCorrectionToResults(self, results):
        print("[{0}] :: GeomAcc - _addSetupCorrectionToResults -- INFO: Adding setup correction to WAD results.".format(os.path.basename(__file__)))

        # Add results of correction for setup
        rotation_in_degrees = np.multiply(self.rigid_transformation_setup[3:6], 180.0 / np.pi)

        results.addFloat("Setup translatie vector lengte", "{:0.1f}".format(np.sqrt(np.sum(np.power(self.rigid_transformation_setup, 2)))))
        self._add3DFloat(results=results, description="Setup rotatie ", value=rotation_in_degrees)
        self._add3DFloat(results=results, description="Setup translatie ", value=self.rigid_transformation_setup)

    def _addNEMAToResults(self, results):
        print("[{0}] :: GeomAcc - _addNEMAToResults -- INFO: Adding NEMA lineariteit to WAD results.".format(os.path.basename(__file__)))

        resultNEMA = self._calculateNEMALinearity()
        results.addFloat("z= +0 NEMA lineariteit", "{:0.2f}".format(resultNEMA))

    def _calculateNEMALinearity(self):

        measurement0 = self.getMeasurement(tablePosition=0)
        detectedPositions = measurement0.correctedMarkerPositions
        closestMarkerIndices = measurement0.closestExpectedMarkerIndices

        NEMALinearity = 0.0

        for pair in self.NEMA_PAIRS:

            pairDistance = np.sqrt(np.sum(np.power(np.subtract(pair[0], pair[1]), 2)))

            p0 = detectedPositions[closestMarkerIndices == measurement0._findMarkerIndex(pair[0])]
            p1 = detectedPositions[closestMarkerIndices == measurement0._findMarkerIndex(pair[1])]

            percentage_line_length_distance = abs(
                100.0 * (1.0 - np.sqrt(np.sum(np.power(np.subtract(p0, p1), 2))) / pairDistance))

            NEMALinearity = np.maximum(NEMALinearity, percentage_line_length_distance)
        return NEMALinearity

    def _addMaxDistanceWithinLimitToResults(self, results):
        for tablePosition in self.getTablePositions():
            curMeasurement = self.getMeasurement(tablePosition)
            resultRadius = curMeasurement.calculateMaxDistanceWithinLimit(limit_in_mm=GeomAccDefaults.LIMITMAXDISTANCE)
            results.addFloat("Max radius within " + str(GeomAccDefaults.LIMITMAXDISTANCE) + " mm for table position " + str(tablePosition), resultRadius)

    def _addLinearityValuesToResults(self, results):
        print("[{0}] :: GeomAcc - _addLinearityValuesToResults -- INFO: Adding linearity to WAD results.".format(os.path.basename(__file__)))
        linearity = self._calculateLinearity()

        results.addFloat("Linearity LR", "{:0.3f}".format(linearity[self.LR]))
        results.addFloat("Linearity AP", "{:0.3f}".format(linearity[self.AP]))
        results.addFloat("Linearity CC", "{:0.3f}".format(linearity[self.CC]))

    def _calculateLinearity(self):
        print("[{0}] :: GeomAcc - _calculateLinearity -- INFO: Calculating linear field gradient.".format(os.path.basename(__file__)))
        detectedPositions = []
        expectedPositions = []
        for tablePosition in self.getTablePositions():
            curMeasurement = self.getMeasurement(tablePosition)

            detectedPositions.extend(curMeasurement.correctedMarkerPositions.tolist())
            expectedPositions.extend(curMeasurement.expectedMarkerPositions[curMeasurement.closestExpectedMarkerIndices].tolist())

        detectedPositions = np.array(detectedPositions)
        expectedPositions = np.array(expectedPositions)

        def linTrans(vec):
            mat = np.diag(vec)
            test_pos = np.dot(detectedPositions, mat)
            diff = test_pos-expectedPositions
            penalty = np.sum(np.power(diff, 2), (0, 1))
            return penalty
        optresult = optimize.minimize(linTrans, [1.0, 1.0, 1.0], bounds=[(.98, 1.02), (.98, 1.02), (.98, 1.02)])

        return optresult.x

    def _addRegionStatisticsToResults(self, results):
        print("[{0}] :: GeomAcc - _addRegionStatisticsToResults -- INFO: Adding region statistics to WAD results.".format(os.path.basename(__file__)))

        for tablePosition in [-130, -60, 0, 60, 130]:
            measurement = self.getMeasurement(tablePosition)
            mean, rms, No_points = measurement.calculateStatisticsForExpectedMarkerPositions(self.BODY_CTR_POSITIONS)

            self._add3DFloat(results, description="z={:+4d} Bodycontour afwijking AVG ".format(int(tablePosition)), value=mean)
            self._add3DFloat(results, description="z={:+4d} Bodycontour afwijking SD ".format(int(tablePosition)), value=rms)

            results.addFloat("z={:+4d} Bodycontour afwijking AVG ".format(int(tablePosition)), "{:0.1f}".format(np.sqrt(np.sum(np.power(mean, 2)))))
            results.addFloat("z={:+4d} Bodycontour afwijking SD ".format(int(tablePosition)), "{:0.1f}".format(np.sqrt(np.sum(np.power(rms, 2)))))

            if No_points > 0:
                results.addFloat("z={:+4d} Aantal gemiste punten - Bodycontour ".format(int(tablePosition)), No_points)

        for tablePosition in self.getTablePositions():
            measurement = self.getMeasurement(tablePosition)
            mean, rms, No_points = measurement.calculateStatisticsForExpectedMarkerPositions(self.CENTER_POSITIONS)

            self._add3DFloat(results, description="z={:+4d} Centrum afwijking AVG ".format(int(tablePosition)), value=mean)
            self._add3DFloat(results, description="z={:+4d} Centrum afwijking SD ".format(int(tablePosition)), value=rms)

            results.addFloat("z={:+4d} Centrum afwijking AVG ".format(int(tablePosition)), "{:0.1f}".format(np.sqrt(np.sum(np.power(mean, 2)))))
            results.addFloat("z={:+4d} Centrum afwijking SD ".format(int(tablePosition)), "{:0.1f}".format(np.sqrt(np.sum(np.power(rms, 2)))))
            if No_points > 0:
                results.addFloat("z={:+4d} Aantal gemiste punten - Centrum afwijking ".format(int(tablePosition)), No_points)

    def save_images_to_wad(self, results):
        print("[{0}] :: GeomAcc - save_images_to_wad -- INFO: Creating and saving figures.".format(os.path.basename(__file__)))

        fileName = "Positions.jpg"
        self.createDeviationFigure(fileName)
        results.addObject("detectedPositions", fileName)

        fileName = "Histograms.jpg"
        self.createHistogramsFigure(fileName)
        results.addObject("Histograms", fileName)

        fileName="Definitions.jpg"
        self.createDefinitionsFigure(fileName)
        results.addObject("Definitions", fileName)

    def createDeviationFigure(self, fileName):
        fig, axs = plt.subplots(ncols=2, nrows=4, sharey=True, sharex=True, figsize=(12, 18))
        title = str(self.studyDate) + " : " + str(self.studyTime) + " " + str(self.studyScanner)
        fig.suptitle(title, fontsize=24)

        self._createDeviationSubplot(ax=axs[0, 0], tablePosition=0)
        self._createDeviationLegend(ax=axs[0, 1])

        self._createDeviationSubplot(ax=axs[1, 0], tablePosition=-60)
        self._createDeviationSubplot(ax=axs[1, 1], tablePosition=60)

        self._createDeviationSubplot(ax=axs[2, 0], tablePosition=-130)
        self._createDeviationSubplot(ax=axs[2, 1], tablePosition=130)

        self._createDeviationSubplot(ax=axs[3, 0], tablePosition=-200)
        self._createDeviationSubplot(ax=axs[3, 1], tablePosition=200)

        fig.tight_layout()
        print("[{0}] :: GeomAcc - createDeviationFigure -- INFO: Saving file.".format(os.path.basename(__file__)))
        plt.subplots_adjust(top=.95)
        fig.savefig(fileName, dpi=160)

    def _createDeviationSubplot(self, ax, tablePosition):

        curMeasurement = self.getMeasurement(tablePosition)
        detectedPositions = curMeasurement.correctedMarkerPositions
        markerPositions = curMeasurement.expectedMarkerPositions
        differences = curMeasurement.differencesCorrectedExpected
        distlength = np.sqrt(np.sum(np.power(differences, 2), axis=1))

        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_title(tablePosition)
        ax.set_xlim(-275, 275)
        ax.set_ylim(-100, 250)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
       
        ax.scatter(markerPositions[:, self.LR], - markerPositions[:,  self.AP], marker='x', c='blue')
        ax.scatter(detectedPositions[distlength > 2, self.LR], - detectedPositions[distlength > 2, self.AP], marker='o', c='red')
        ax.scatter(detectedPositions[distlength < 2, self.LR], - detectedPositions[distlength < 2, self.AP], marker='o', c='yellow')
        ax.scatter(detectedPositions[distlength < 1, self.LR], - detectedPositions[distlength < 1, self.AP], marker='o', c='green')

    def _createDeviationLegend(self, ax):

        blue_cross = mlines.Line2D([], [], color='blue', marker='*', markersize=15, linestyle=None, label='Expected marker')
        green = mlines.Line2D([], [], color='green', marker='o', markersize=15, linestyle=None, label=r'$\delta$ < 1 mm')
        yellow = mlines.Line2D([], [], color='yellow', linestyle=None, marker='o', markersize=15, label=r'1 mm < $\delta$ < 2 mm')
        red = mlines.Line2D([], [], color='red', linestyle=None, marker='o', markersize=15, label=r'$\delta$ > 2 mm')
        ax.legend(handles=[blue_cross, green, yellow, red], loc=10, title=r'Deviation $\delta$ from expected position')
        ax.set_axis_off()

    def createHistogramsFigure(self, fileName):
        fig, axs = plt.subplots(ncols=2, nrows=4, sharey=True, sharex=True, figsize=(12, 18))
        title = str(self.studyDate) + " : " + str(self.studyTime) + " " + str(self.studyScanner)
        fig.suptitle(title, fontsize=24)
        fig.subplots_adjust(top=1)
        self._createHistogramPlot(ax=axs[0, 0], tablePosition=0)
        self._createHistogramLegend(ax=axs[0, 1])

        self._createHistogramPlot(ax=axs[1, 0], tablePosition=-60)
        self._createHistogramPlot(ax=axs[1, 1], tablePosition=60)

        self._createHistogramPlot(ax=axs[2, 0], tablePosition=-130)
        self._createHistogramPlot(ax=axs[2, 1], tablePosition=130)

        self._createHistogramPlot(ax=axs[3, 0], tablePosition=-200)
        axs[3, 0].set_xlabel('mm')
        self._createHistogramPlot(ax=axs[3, 1], tablePosition=200)
        axs[3, 1].set_xlabel('mm')

        fig.tight_layout()
        print("[{0}] :: GeomAcc - createHistogramsFigure -- INFO: Saving file.".format(os.path.basename(__file__)))
        plt.subplots_adjust(top=.95)
        fig.savefig(fileName, dpi=160)

    def _createHistogramPlot(self, ax, tablePosition):
        curMeasurement = self.getMeasurement(tablePosition)
        differences = curMeasurement.differencesCorrectedExpected

        bins = np.linspace(start=-3, stop=3, num=13)

        mu_AP, mu_LR, mu_CC = np.mean(differences, axis=0)
        sigma_AP, sigma_LR, sigma_CC = np.sqrt(np.mean(np.power(differences, 2), axis=0))
        textstr = '$\mu=(%.1f, %.1f, %.1f)$ mm \n$\sigma=(%.1f, %.1f, %.1f)$ mm' % (mu_AP, mu_LR, mu_CC, sigma_AP, sigma_LR, sigma_CC)
        textstrprops = dict(boxstyle='round', facecolor='white', alpha=0.5)

        ax.set_facecolor('white')
        ax.set_title(tablePosition)
        ax.hist((differences[:, self.AP], differences[:, self.LR], differences[:, self.CC]), bins=bins, normed=False, label=('AP', 'LR', 'CC'), color=['green', 'blue', 'red'], rwidth=0.66)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=textstrprops)

    def _createHistogramLegend(self, ax):
        green = mpatches.Patch(color='green', label='AP')
        blue = mpatches.Patch(color='blue', label='LR')
        red = mpatches.Patch(color='red', label='CC')

        ax.legend(handles=[green, blue, red], loc=10, title=r'Deviation $\delta$ from expected position')
        ax.set_axis_off()

    def createDefinitionsFigure(self, fileName):
        fig, axs = plt.subplots(ncols=2, nrows=4, sharey=True, sharex=False, figsize=(12, 18))
        title = str(self.studyDate) + " : " + str(self.studyTime) + " " + str(self.studyScanner)
        fig.suptitle(title, fontsize=24)
        fig.subplots_adjust(top=1)
        markerPositions = self.getMeasurement(tablePosition=0).expectedMarkerPositions

        #NEMA DEFINITION
        ax = axs[0, 0]
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_title('NEMA')
        ax.set_xlim(-275, 275)
        ax.set_ylim(-100, 250)

        ax.scatter(markerPositions[:, self.LR], - markerPositions[:, self.AP], marker='x', color='blue')
        for pt1, pt2 in self.NEMA_PAIRS:
            ax.plot((pt1[self.LR], pt2[self.LR]), (-pt1[self.AP], -pt2[self.AP]), color='white')
        ax.set_xlabel('LR [mm]')
        ax.set_ylabel('AP [mm]')

        # BODY and CENTER DEFINITIONS
        ax = axs[0, 1]
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_title('Regions')
        ax.set_xlim(-275, 275)
        ax.set_ylim(-100, 250)
        ax.scatter(markerPositions[:, self.LR], - markerPositions[:, self.AP], marker='x', color='blue')
        ax.scatter(self.CENTER_POSITIONS[:, self.LR], - self.CENTER_POSITIONS[:, self.AP], marker='o', c='green', label='Center')
        ax.scatter(self.BODY_CTR_POSITIONS[:, self.LR], - self.BODY_CTR_POSITIONS[:, self.AP], marker='o', c='purple', label='Body')

        ax.set_xlabel('LR [mm]')
        ax.set_ylabel('AP [mm]')
        ax.legend()

        ax = axs[1, 0]
        ax.set_axis_off()
        ax = axs[1, 1]
        ax.set_axis_off()
        ax = axs[2, 0]
        ax.set_axis_off()
        ax = axs[2, 1]
        ax.set_axis_off()
        ax = axs[3, 0]
        ax.set_axis_off()
        ax = axs[3, 1]
        ax.set_axis_off()

        fig.tight_layout()
        print("[{0}] :: GeomAcc - createDefinitionsFigure -- INFO: Saving file.".format(os.path.basename(__file__)))
        plt.subplots_adjust(top=.95)
        fig.savefig(fileName, dpi=160)

    def _add3DFloat(self, results, description, value):
        results.addFloat(description + "LR", "{:0.1f}".format(value[self.LR]))
        results.addFloat(description + "AP", "{:0.1f}".format(value[self.AP]))
        results.addFloat(description + "CC", "{:0.1f}".format(value[self.CC]))