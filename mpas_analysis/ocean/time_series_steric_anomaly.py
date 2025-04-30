# -*- coding: utf-8 -*-
# This software is open source software available under the BSD-3 license.
#
# Copyright (c) 2025 Triad National Security, LLC. All rights reserved.
# Copyright (c) 2025 Lawrence Livermore National Security, LLC. All rights
# reserved.
# Copyright (c) 2025 UT-Battelle, LLC. All rights reserved.
#
# Additional copyright and license information can be found in the LICENSE file
# distributed with this code, or at
# https://raw.githubusercontent.com/MPAS-Dev/MPAS-Analysis/main/LICENSE
#
import cftime
import numpy
import os
import matplotlib.pyplot as plt
import xarray as xr

from mpas_analysis.shared import AnalysisTask
from mpas_analysis.shared.generalized_reader import open_multifile_dataset

from mpas_analysis.shared.io import open_mpas_dataset
from mpas_analysis.shared.io.utility import build_config_full_path
from mpas_analysis.shared.io.utility import build_config_full_path, \
    make_directories
from mpas_analysis.shared.plot import timeseries_analysis_plot, savefig
from mpas_analysis.shared.html import write_image_xml
from mpas_analysis.shared.timekeeping.utility import get_simulation_start_time
from mpas_analysis.ocean.sealevel.steric import calc_steric


class TimeSeriesStericAnomaly(AnalysisTask):
    """
    Plots a time series of the global mean steric sea level relative to a
    reference time.

    Attributes
    ----------
    timeSeriesFileName : str
        The name of the file where the steric sea-level anomaly is stored

    controlConfig : mpas_tools.config.MpasConfigParser
        Configuration options for a control run (if one is provided)

    filePrefix : str
        The basename (without extension) of the PNG and XML files to write out
    """
    # Authors
    # -------
    # Xylar Asay-Davis, Matthew Hoffman

    def __init__(self, config, controlConfig):

        """
        Construct the analysis task.

        Parameters
        ----------
        config : mpas_tools.config.MpasConfigParser
            Configuration options

        controlConfig : mpas_tools.config.MpasConfigParser
            Configuration options for a control run (if any)
        """
        # Authors
        # -------
        # Xylar Asay-Davis, Matthew Hoffman

        # first, call the constructor from the base class (AnalysisTask)
        super().__init__(
            config=config,
            taskName='timeSeriesStericAnomaly',
            componentName='ocean',
            tags=['timeSeries', 'steric', 'publicObs', 'anomaly'])

        self.controlConfig = controlConfig
        self.filePrefix = None

        self.timeSeriesFileName = 'globalMeanStericAnomaly.nc'


    def setup_and_check(self):
        """
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        OSError
            If files are not present
        """
        # Authors
        # -------
        # Xylar Asay-Davis, Matthew Hoffman

        # first, call setup_and_check from the base class (AnalysisTask),
        # which will perform some common setup, including storing:
        #   self.inDirectory, self.plotsDirectory, self.namelist, self.streams
        #   self.calendar
        super().setup_and_check()

        config = self.config


        self.xmlFileNames = []

        mainRunName = config.get('runs', 'mainRunName')
        self.filePrefix = 'steric_anomaly_global_{}'.format(mainRunName)
        self.xmlFileNames.append('{}/{}.xml'.format(self.plotsDirectory,
                                                    self.filePrefix))

        self.restartFileName = self.runStreams.readpath('restart')[0]

        baseDirectory = build_config_full_path(
            config, 'output', 'timeSeriesSubdirectory')

        make_directories(baseDirectory)

        # get a list of timeSeriesStats output files from the streams file,
        # reading only those that are between the start and end dates
        self.startYear = config.getint('timeSeries', 'startYear')
        self.endYear = config.getint('timeSeries', 'endYear')
        self.startDate = '{:04d}-01-01_00:00:00'.format(self.startYear)
        self.endDate = '{:04d}-12-31_23:59:59'.format(self.endYear)
        streamName = 'timeSeriesStatsMonthlyOutput'
        self.inputFiles = self.historyStreams.readpath(
            streamName, startDate=self.startDate, endDate=self.endDate,
            calendar=self.calendar)

        if len(self.inputFiles) == 0:
            raise IOError('No files were found in stream {} between {} and '
                          '{}.'.format(streamName, startDate, endDate))

        self.runMessage = \
            f'\nComputing MPAS time series from first year plus files:\n' \
            f'    {os.path.basename(self.inputFiles[0])} through\n' \
            f'    {os.path.basename(self.inputFiles[-1])}'

        # Make sure first year of data is included for computing anomalies
        if config.has_option('timeSeries', 'anomalyRefYear'):
            anomalyYear = config.getint('timeSeries', 'anomalyRefYear')
            anomalyStartDate = '{:04d}-01-01_00:00:00'.format(anomalyYear)
        else:
            anomalyStartDate = get_simulation_start_time(self.runStreams)
            anomalyYear = int(anomalyStartDate[0:4])

        anomalyEndDate = '{:04d}-12-31_23:59:59'.format(anomalyYear)
        firstYearInputFiles = self.historyStreams.readpath(
            streamName, startDate=anomalyStartDate,
            endDate=anomalyEndDate,
            calendar=self.calendar)
        for fileName in firstYearInputFiles:
            if fileName not in self.inputFiles:
                self.inputFiles.append(fileName)

        self.inputFiles = sorted(self.inputFiles)

        with xr.open_dataset(self.inputFiles[0],  # TODO: should this be replaced with open_mpas_dataset?
                             decode_timedelta=False) as ds:
            self.allVariables = list(ds.data_vars.keys())


    def run_task(self):
        """
        Performs analysis of the time-series output of steric sea level
        """
        # Authors
        # -------
        # Xylar Asay-Davis, Matthew Hoffman

        self.logger.info("\nPlotting time series of steric anomaly...")

        self.logger.info('  Load steric anomaly data...')
        
        config = self.config
        calendar = self.calendar

        ds_mesh = xr.open_dataset(self.restartFileName)
        areacello = ds_mesh['areaCell'].load()
        areacello.name = 'areacello'
        latCell = ds_mesh['latCell'].load()
        lonCell = ds_mesh['lonCell'].load()

        startDate = self.config.get('timeSeries', 'startDate')
        endDate = self.config.get('timeSeries', 'endDate')
        streamName = 'timeSeriesStatsMonthlyOutput'
        self.inputFiles = self.historyStreams.readpath(
            streamName, startDate=startDate, endDate=endDate,
            calendar=self.calendar)
        #open_multifile_dataset(self.inputFiles, self.calendar, self.config)
        ds = xr.open_mfdataset(self.inputFiles,
                               combine='nested', concat_dim='Time',
                               decode_timedelta=False)

        # set up data structure
        so = ds['timeMonthly_avg_activeTracers_salinity']
        so.name = 'so'
        thetao = ds['timeMonthly_avg_activeTracers_temperature']
        thetao.name = 'thetao'
        dz = ds['timeMonthly_avg_layerThickness']
        dz.name = 'dz'
        volcello = dz * areacello
        volcello.name = 'volcello'
        rho = ds['timeMonthly_avg_density']
        rho.name = 'rho'
        ds_input = xr.merge([so, thetao, areacello, volcello, rho, dz])
        nt = ds_input.sizes['Time']
        print(f'nt={nt}')

        # calculate global steric SL
        resultg, reference = calc_steric(ds_input, domain='global')
        resultg.load()
        print(f'GLOBAL: ref height={resultg["reference_height"].values}, sealevel={resultg["steric"].values}')

        baseDirectory = build_config_full_path(
            config, 'output', 'timeSeriesSubdirectory')
        fileName = '{}/{}'.format(baseDirectory, self.timeSeriesFileName)
        resultg.to_netcdf(fileName)

        if self.controlConfig is not None:
            baseDirectory = build_config_full_path(
                self.controlConfig, 'output', 'timeSeriesSubdirectory')

            controlFileName = '{}/{}'.format(baseDirectory,
                                             self.timeSeriesFileName)

            controlStartYear = self.controlConfig.getint(
                'timeSeries', 'startYear')
            controlEndYear = self.controlConfig.getint('timeSeries', 'endYear')
            controlStartDate = '{:04d}-01-01_00:00:00'.format(controlStartYear)
            controlEndDate = '{:04d}-12-31_23:59:59'.format(controlEndYear)

            dsRef = open_mpas_dataset(
                fileName=controlFileName,
                calendar=calendar,
                variableList=variableList,
                timeVariableNames=None,
                startDate=controlStartDate,
                endDate=controlEndDate)
        else:
            dsRef = None

        self.logger.info('  Make plots...')
        title = 'Global Mean Steric Sea-Level Anomaly'
        xLabel = 'Time (years)'
        yLabel = 'Sea-level change (m)'
        mainRunName = config.get('runs', 'mainRunName')

        resultg['Time'] = cftime.date2num(resultg.Time,
                                          f'days since {self.startDate}',
                                          calendar=self.calendar)
        steric = resultg.steric

        outFileName = '{}/{}.png'.format(self.plotsDirectory, self.filePrefix)

        lineColors = [config.get('timeSeries', 'mainColor')]
        lineWidths = [3]

        fields = [steric]
        legendText = [mainRunName]

        if dsRef is not None:
            refSteric = dsRef.steric
            fields.append(refSteric)
            lineColors.append(config.get('timeSeries', 'controlColor'))
            lineWidths.append(1.5)
            controlRunName = self.controlConfig.get('runs', 'mainRunName')
            legendText.append(controlRunName)

        if config.has_option(self.taskName, 'firstYearXTicks'):
            firstYearXTicks = config.getint(self.taskName,
                                            'firstYearXTicks')
        else:
            firstYearXTicks = None

        if config.has_option(self.taskName, 'yearStrideXTicks'):
            yearStrideXTicks = config.getint(self.taskName,
                                             'yearStrideXTicks')
        else:
            yearStrideXTicks = None

        timeseries_analysis_plot(config, fields, calendar=calendar,
                                 title=title, xlabel=xLabel, ylabel=yLabel,
                                 lineColors=lineColors,
                                 lineWidths=lineWidths,
                                 legendText=legendText,
                                 firstYearXTicks=firstYearXTicks,
                                 yearStrideXTicks=yearStrideXTicks)

        if config.has_option(self.taskName, 'fitStartYear') and \
                config.has_option(self.taskName, 'fitEndYear'):
            fitStartYear = config.getint(self.taskName, 'fitStartYear')
            fitEndYear = config.getint(self.taskName, 'fitEndYear')
            fitStartDate = '{:04d}-01-01_00:00:00'.format(fitStartYear)
            fitEndDate = '{:04d}-12-31_23:59:59'.format(fitEndYear)

            dsFit = open_mpas_dataset(fileName=fileName, calendar=calendar,
                                      variableList=variableList,
                                      timeVariableNames=None,
                                      startDate=fitStartDate,
                                      endDate=fitEndDate)

            time = dsFit.Time.values
            deltaSSH = dsFit[varName].values

            coeffs = numpy.polyfit(time, deltaSSH, deg=1)
            print(coeffs)
            line = numpy.poly1d(coeffs)
            ylim = plt.gca().get_ylim()

            # m/day --> m/yr
            slope = coeffs[0]*365

            label = r'linear fit {:04d}-{:04d}: slope = ${}$ m/yr'.format(
                fitStartYear, fitEndYear, _as_si(slope, 2))

            fitColor = config.get('timeSeries', 'fitColor1')
            full_time = ds.Time.values
            plt.plot(full_time, line(full_time), color=fitColor,
                     linewidth=0.6*lineWidths[0],
                     label=label)

            plt.plot([time[0], time[-1]], line([time[0], time[-1]]), '.',
                     color=fitColor, markersize=16)
            plt.gca().set_ylim(ylim)
            plt.legend(loc='best')

        savefig(outFileName, config)

        caption = title
        write_image_xml(
            config=config,
            filePrefix=self.filePrefix,
            componentName='Ocean',
            componentSubdirectory='ocean',
            galleryGroup='Time Series',
            groupLink='timeseries',
            thumbnailDescription='Steric Anomaly',
            imageDescription=caption,
            imageCaption=caption)


def _as_si(x, ndp):
    """
    https://stackoverflow.com/a/31453961/7728169
    """
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
