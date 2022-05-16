"""Main script for the FTpyR user interface."""
import os
import sys
import yaml
import time
import logging
import warnings
import traceback
import qdarktheme
import numpy as np
import pandas as pd
from scipy import odr
from functools import partial
from scipy.optimize import curve_fit
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThreadPool, QRunnable
from PySide6.QtWidgets import (QMainWindow, QScrollArea, QGridLayout,
                               QApplication, QToolBar, QFrame, QSplitter,
                               QProgressBar, QLabel, QLineEdit, QTextEdit,
                               QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
                               QPlainTextEdit, QPushButton, QFileDialog,
                               QWidget, QTabWidget, QDialog, QTableWidget,
                               QTableWidgetItem, QMenu, QHeaderView)
import pyqtgraph as pg
import pyqtgraph.dockarea as da

from ftpyr.read import read_spectrum
from ftpyr.analyse import Analyser
from ftpyr.parameters import Parameters


__version__ = '0.1.0'
__author__ = 'Ben Esse'

# =============================================================================
# =============================================================================
# Setup logging
# =============================================================================
# =============================================================================

# Connect to the logger
logger = logging.getLogger()
warnings.filterwarnings('ignore')


class Signaller(QObject):
    """Signaller object for logging from QThreads."""
    signal = Signal(str, logging.LogRecord)


class QtHandler(logging.Handler):
    """logging Handler object for handling logs from QThreads."""

    def __init__(self, slotfunc, *args, **kwargs):
        super(QtHandler, self).__init__(*args, **kwargs)
        self.signaller = Signaller()
        self.signaller.signal.connect(slotfunc)

    def emit(self, record):
        s = self.format(record)
        self.signaller.signal.emit(s, record)


# =============================================================================
# =============================================================================
# Main GUI Window
# =============================================================================
# =============================================================================

class MainWindow(QMainWindow):
    """Main GUI window."""

    # Set log level colors
    LOGCOLORS = {
        logging.DEBUG: 'darkgrey',
        logging.INFO: 'darkgrey',
        logging.WARNING: 'orange',
        logging.ERROR: 'red',
        logging.CRITICAL: 'purple',
    }

    # Set default plot colors
    PLOTCOLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Grey
        '#bcbd22',  # Lime
        '#17becf'   # Lightblue
    ]

    def __init__(self, app, *args, **kwargs):
        """Initialise the main window."""
        super(MainWindow, self).__init__(*args, **kwargs)
        self.app = app

        # Main Window Setup ===================================================

        # Set the window properties
        self.setWindowTitle(f'FTpyR {__version__}')
        self.statusBar().showMessage('Ready')
        self.setGeometry(40, 40, 1210, 700)
        self.setWindowIcon(QIcon('bin/icons/main.ico'))

        # Set the window layout
        self.generalLayout = QGridLayout()
        self._centralWidget = QScrollArea()
        self.widget = QFrame()
        self.setCentralWidget(self._centralWidget)
        self.widget.setLayout(self.generalLayout)

        # Scroll Area Properties
        self._centralWidget.setWidgetResizable(True)
        self._centralWidget.setWidget(self.widget)

        # Set the default theme
        self.theme = 'Dark'

        self.threadpool = QThreadPool()

        # Global actions ======================================================

        # Save action
        saveAct = QAction(QIcon('bin/icons/save.png'), '&Save', self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.triggered.connect(partial(self.saveConfig, False))

        # Save As action
        saveasAct = QAction(QIcon('bin/icons/saveas.png'), '&Save As', self)
        saveasAct.setShortcut('Ctrl+Shift+S')
        saveasAct.triggered.connect(partial(self.saveConfig, True))

        # Load action
        loadAct = QAction(QIcon('bin/icons/open.png'), '&Load', self)
        loadAct.triggered.connect(partial(self.loadConfig, None))

        # Change theme action
        themeAct = QAction(QIcon('bin/icons/theme.png'), '&Change Theme', self)
        themeAct.triggered.connect(self.changeTheme)

        # Add new window action
        newWindowAct = QAction(QIcon('bin/icons/add.png'), '&New Window', self)
        newWindowAct.triggered.connect(self.generateNewWindow)

        # Add menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveAct)
        fileMenu.addAction(saveasAct)
        fileMenu.addAction(loadAct)
        toolMenu = menubar.addMenu('&View')
        toolMenu.addAction(themeAct)
        toolMenu = menubar.addMenu('&Tools')
        toolMenu.addAction(newWindowAct)

        # Create a toolbar
        toolBar = QToolBar("Main toolbar")
        self.addToolBar(toolBar)
        toolBar.addAction(saveAct)
        toolBar.addAction(saveasAct)
        toolBar.addAction(loadAct)
        toolBar.addAction(themeAct)
        toolBar.addAction(newWindowAct)

        # Layout frames =======================================================

        # Create a frame to control the inputs
        self.inputFrame = QFrame()
        self.inputFrame.setFrameShape(QFrame.StyledPanel)

        # Create a frame for the program output log
        self.logFrame = QFrame()
        self.logFrame.setFrameShape(QFrame.StyledPanel)

        # Create a frame for the setup and graphs
        self.outputFrame = QFrame()
        self.outputFrame.setFrameShape(QFrame.StyledPanel)

        # Add splitters to allow for adjustment
        splitter1 = QSplitter(Qt.Vertical)
        splitter1.insertWidget(0, self.inputFrame)
        splitter1.insertWidget(1, self.logFrame)

        splitter2 = QSplitter(Qt.Horizontal)
        splitter2.insertWidget(0, splitter1)
        splitter2.insertWidget(1, self.outputFrame)

        # Pack the Frames and splitters
        self.generalLayout.addWidget(splitter2)

        # Generate control widgets ============================================

        # Create an empty dictionary to hold the GUI widgets
        self.widgets = Widgets()

        # Create holders for the tabs, graphwindows and window data
        self.inputTabs = {}
        self.outputTabs = {}
        self.windowWidgets = {}
        self.windows = []
        self.plot_axes = {}
        self.plot_lines = {}
        self.plot_regions = {}
        self.results_tables = {}
        self.species_list = {}
        self.fit_results = {}

        # Read in gas list
        with open('databases/atm_layer.yml', 'r') as ymlfile:
            self.gas_list = yaml.load(ymlfile, Loader=yaml.FullLoader)

        # Generate the GUI widgets
        self._createInputs()
        self._createLogs()
        self._createOutputs()

        # Apply theme and read in settings ====================================

        # Update widgets from loaded config file
        self.config = {}
        self.config_fname = None
        if os.path.isfile('bin/.config'):
            with open('bin/.config', 'r') as r:
                self.config_fname = r.readline().strip()
            self.loadConfig(fname=self.config_fname)

        # Update GUI theme
        if self.theme == 'Dark':
            self.theme = 'Light'
        elif self.theme == 'Light':
            self.theme = 'Dark'
        self.changeTheme()

    # =========================================================================
    # Build GUI widgets
    # =========================================================================

    def _createInputs(self):
        """Generate input widgets."""
        layout = QGridLayout(self.inputFrame)
        layout.setAlignment(Qt.AlignTop)

        # Form the tab widget
        self.inputTabHolder = QTabWidget()
        layout.addWidget(self.inputTabHolder, 0, 0)

        # File selection ======================================================

        globalTabs = QTabWidget()
        self.inputTabHolder.addTab(globalTabs, 'Global Setup')

        # Add tabs
        fileTab = QWidget()
        globalTabs.addTab(fileTab, 'File Setup')

        # Add layout
        file_layout = QGridLayout(fileTab)
        file_layout.setAlignment(Qt.AlignTop)

        # Add an input for the save selection
        file_layout.addWidget(QLabel('Output\nFolder:'), 0, 0)
        self.widgets['save_dir'] = QLineEdit('Results')
        self.widgets['save_dir'].setToolTip('Folder to hold results')
        file_layout.addWidget(self.widgets['save_dir'], 0, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.setFixedSize(70, 25)
        btn.clicked.connect(
            partial(self.browse, self.widgets['save_dir'], 'folder', None)
        )
        file_layout.addWidget(btn, 0, 4)

        # Add an input for the spectra selection
        file_layout.addWidget(QLabel('Spectra:'), 1, 0)
        self.widgets['spec_fnames'] = QTextEdit()
        self.widgets['spec_fnames'].setToolTip('Measurement spectrum files')
        file_layout.addWidget(self.widgets['spec_fnames'], 1, 1, 1, 3)
        self.widgets['spec_fnames'].textChanged.connect(
            self.plot_first_spectrum
        )
        self.first_filename = ''
        btn = QPushButton('Browse')
        btn.setFixedSize(70, 25)
        btn.clicked.connect(
            partial(self.browse, self.widgets['spec_fnames'], 'multi', None)
        )
        file_layout.addWidget(btn, 1, 4)

        # # Add an input for the background spectrum
        # file_layout.addWidget(QLabel('Background\nSpectrum:'), 2, 0)
        # self.widgets['bg_fname'] = QLineEdit()
        # file_layout.addWidget(self.widgets['bg_fname'], 2, 1, 1, 3)
        # btn = QPushButton('Browse')
        # btn.setFixedSize(70, 25)
        # btn.clicked.connect(
        #     partial(self.browse, self.widgets['bg_fname'], 'single', None)
        # )
        # file_layout.addWidget(btn, 2, 4)
        #
        # # Add control for background correction
        # file_layout.addWidget(QLabel('Background\nBehaviour:'), 3, 0)
        # self.widgets['bg_behaviour'] = QComboBox()
        # self.widgets['bg_behaviour'].addItems(
        #     ['ignore', 'subtract', 'divide'])
        # self.widgets['bg_behaviour'].setFixedSize(100, 20)
        # file_layout.addWidget(self.widgets['bg_behaviour'], 3, 1)

        # Add an input for the RFM exe path
        file_layout.addWidget(QLabel('RFM:'), 4, 0)
        self.widgets['rfm_path'] = QLineEdit()
        self.widgets['rfm_path'].setToolTip('RFM executable file')
        file_layout.addWidget(self.widgets['rfm_path'], 4, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.setFixedSize(70, 25)
        btn.clicked.connect(
            partial(self.browse, self.widgets['rfm_path'], 'single', None)
        )
        file_layout.addWidget(btn, 4, 4)

        # Add an input for the HITRAN database path
        file_layout.addWidget(QLabel('HITRAN:'), 5, 0)
        self.widgets['hitran_path'] = QLineEdit()
        self.widgets['hitran_path'].setToolTip('HITRAN database file')
        file_layout.addWidget(self.widgets['hitran_path'], 5, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.setFixedSize(70, 25)
        btn.clicked.connect(
            partial(self.browse, self.widgets['hitran_path'], 'single', None)
        )
        file_layout.addWidget(btn, 5, 4)

        # Spectrometer setup ==================================================

        # Add tab
        specTab = QWidget()
        globalTabs.addTab(specTab, 'Model')

        # Add layout
        spec_layout = QGridLayout(specTab)
        spec_layout.setAlignment(Qt.AlignTop)

        # Input for apodization function
        spec_layout.addWidget(QRightLabel('Apodization\nFunction'), 0, 0)
        self.widgets['apod_function'] = QComboBox()
        self.widgets['apod_function'].addItems(
            ['NB_weak', 'NB_medium', 'NB_strong', 'triangular', 'boxcar']
        )
        self.widgets.set('apod_function', 'NB_medium')
        spec_layout.addWidget(self.widgets['apod_function'], 0, 1)

        # Input for fov
        spec_layout.addWidget(QRightLabel('Field of View\n(m.rad)'), 0, 2)
        self.widgets['fov'] = DSpinBox(10, [0, 1000], 1.0)
        spec_layout.addWidget(self.widgets['fov'], 0, 3)
        self.widgets['fit_fov'] = QCheckBox('Fit?')
        self.widgets['fit_fov'].setChecked(False)
        spec_layout.addWidget(self.widgets['fit_fov'], 0, 4)

        # Input for zero filling
        spec_layout.addWidget(QRightLabel('Zero Filling\nFactor'), 1, 0)
        self.widgets['zero_fill_factor'] = SpinBox(0, [0, 100])
        spec_layout.addWidget(self.widgets['zero_fill_factor'], 1, 1)

        # Input for OPD
        spec_layout.addWidget(
            QRightLabel('Optical Path\nDifference (cm)'), 1, 2)
        self.widgets['opd'] = DSpinBox(1.6, [0, 100], 0.01)
        spec_layout.addWidget(self.widgets['opd'], 1, 3)

        # New row
        spec_layout.addWidget(QHLine(), 2, 0, 1, 10)

        # Add control for updating parameters
        self.widgets['solar_flag'] = QCheckBox('Solar\nOccultation')
        spec_layout.addWidget(self.widgets['solar_flag'], 3, 0)

        # Add control for maximum residual
        spec_layout.addWidget(QRightLabel('Observation\nHeight (m)'), 3, 1)
        self.widgets['obs_height'] = DSpinBox(0, [0, 100000], 0.1)
        spec_layout.addWidget(self.widgets['obs_height'], 3, 2)
        self.widgets['solar_flag'].stateChanged.connect(
            lambda: self.widgets['obs_height'].setDisabled(
                not self.widgets['solar_flag'].isChecked()
            )
        )
        self.widgets['solar_flag'].setChecked(False)

        # New row
        spec_layout.addWidget(QHLine(), 4, 0, 1, 10)

        # Add control for updating parameters
        self.widgets['update_params'] = QCheckBox('Update Fit\nParameters?')
        spec_layout.addWidget(self.widgets['update_params'], 5, 0)

        # Add control for maximum residual
        spec_layout.addWidget(QRightLabel('Good Fit\nResidual Limit'), 5, 1)
        self.widgets['residual_limit'] = DSpinBox(10, [0, 1000], 0.1)
        spec_layout.addWidget(self.widgets['residual_limit'], 5, 2)
        self.widgets['update_params'].stateChanged.connect(
            lambda: self.widgets['residual_limit'].setDisabled(
                not self.widgets['update_params'].isChecked()
            )
        )
        self.widgets['update_params'].setChecked(False)

        # New row
        spec_layout.addWidget(QHLine(), 6, 0, 1, 10)

        spec_layout.addWidget(QRightLabel('Output Units:'), 7, 0)
        self.widgets['output_units'] = QComboBox()
        self.widgets['output_units'].addItems(['molecules.cm-2', 'ppm.m'])
        spec_layout.addWidget(self.widgets['output_units'], 7, 1)

    def _createLogs(self):
        """Generate program log and control widgets."""
        layout = QGridLayout(self.logFrame)
        layout.setAlignment(Qt.AlignTop)

        # Add button to begin analysis
        self.start_btn = QPushButton('Begin!')
        self.start_btn.setToolTip('Begin spectra analysis')
        self.start_btn.clicked.connect(self.control_loop)
        self.start_btn.setFixedSize(90, 25)
        layout.addWidget(self.start_btn, 0, 1)

        # Add button to pause analysis
        self.pause_btn = QPushButton('Pause')
        self.pause_btn.setToolTip('Pause/play spectra analysis')
        self.pause_btn.clicked.connect(partial(self.pause_analysis))
        self.pause_btn.setFixedSize(90, 25)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn, 0, 2)

        # Add button to stop analysis
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setToolTip('Stop spectra analysis')
        self.stop_btn.clicked.connect(partial(self.stop_analysis))
        self.stop_btn.setFixedSize(90, 25)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn, 0, 3)

        # Add a progress bar
        self.progress = QProgressBar(self)
        layout.addWidget(self.progress, 1, 0, 1, 5)

        # Create a textbox to display the program logs
        self.logBox = QPlainTextEdit(self)
        self.logBox.setReadOnly(True)
        formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')
        self.handler = QtHandler(self.updateLog)
        self.handler.setFormatter(formatter)
        logger.addHandler(self.handler)
        logger.setLevel(logging.INFO)
        layout.addWidget(self.logBox, 2, 0, 1, 5)
        logger.info(f'Welcome to FTpyR v{__version__}!')

    def _createOutputs(self):
        """Generate output and display widgets."""
        layout = QGridLayout(self.outputFrame)
        layout.setAlignment(Qt.AlignTop)

        # Form the tab widget
        self.outputTabHolder = QTabWidget()
        layout.addWidget(self.outputTabHolder, 0, 0)

        # Setup the initial results graph view ================================

        graphTab = QWidget()
        self.outputTabHolder.addTab(graphTab, "Overview")
        layout = QGridLayout(graphTab)
        spec_area = da.DockArea()
        layout.addWidget(spec_area, 0, 0)

        # Generate the docks
        spec_dock = da.Dock('Spectrum')
        spec_area.addDock(spec_dock, 'top')

        # Generate axes
        ax0 = pg.PlotWidget()
        spec_dock.addWidget(ax0)

        # Generate plot lines
        pen0 = pg.mkPen(color=self.PLOTCOLORS[0], width=0.8)
        l0 = ax0.plot(pen=pen0)

        # Add axes labels
        ax0.setLabel('left', 'Intensity (counts)')
        ax0.setLabel('bottom', 'Wavenumber (cm-1)')

        # Ratio graph =========================================================

        ratioTab = QWidget()
        self.outputTabHolder.addTab(ratioTab, "Ratios")
        layout = QGridLayout(ratioTab)
        layout.setAlignment(Qt.AlignTop)

        # Generate menus to select ratio species
        layout.addWidget(QRightLabel('x-Window'), 0, 0)
        self.widgets['ratio_WindowX'] = QComboBox()
        self.widgets['ratio_WindowX'].currentTextChanged.connect(
            lambda: self.update_ratio_combo('x')
        )
        layout.addWidget(self.widgets['ratio_WindowX'], 0, 1)
        layout.addWidget(QRightLabel('x-Species'), 1, 0)
        self.widgets['ratioSpeciesX'] = QComboBox()
        layout.addWidget(self.widgets['ratioSpeciesX'], 1, 1)
        self.widgets['ratioSpeciesX'].currentTextChanged.connect(
            self.update_ratio_plot
        )
        layout.addWidget(QRightLabel('y-Window'), 0, 2)
        self.widgets['ratio_WindowY'] = QComboBox()
        self.widgets['ratio_WindowY'].currentTextChanged.connect(
            lambda: self.update_ratio_combo('y')
        )
        layout.addWidget(self.widgets['ratio_WindowY'], 0, 3)
        layout.addWidget(QRightLabel('y-Species'), 1, 2)
        self.widgets['ratioSpeciesY'] = QComboBox()
        layout.addWidget(self.widgets['ratioSpeciesY'], 1, 3)
        self.widgets['ratioSpeciesY'].currentTextChanged.connect(
            self.update_ratio_plot
        )

        # Add displays for the fitted gradient and intercept
        layout.addWidget(QRightLabel('Gradient:'), 0, 4)
        self.ratio_gradient = QLabel('-')
        layout.addWidget(self.ratio_gradient, 0, 5)
        layout.addWidget(QRightLabel('Intercept:'), 1, 4)
        self.ratio_intercept = QLabel('-')
        layout.addWidget(self.ratio_intercept, 1, 5)

        # Add option to include errors in fit
        self.widgets['ratio_fit_errors'] = QCheckBox('Include\nErrors?')
        layout.addWidget(self.widgets['ratio_fit_errors'], 0, 6)
        self.widgets['ratio_fit_errors'].stateChanged.connect(
            self.update_ratio_plot)

        # Add option to remove bad fits
        self.widgets['bad_fit_flag'] = QCheckBox('Remove\nBad Fits?')
        layout.addWidget(self.widgets['bad_fit_flag'], 1, 6)
        self.widgets['bad_fit_flag'].stateChanged.connect(
            self.update_ratio_plot)

        # Generate the plot
        ratio_area = da.DockArea()
        layout.addWidget(ratio_area, 2, 0, 1, 7)

        # Generate the docks
        ratio_dock = da.Dock('Ratio')
        ratio_area.addDock(ratio_dock, 'top')

        # Generate axes
        ax1 = pg.PlotWidget()
        ratio_dock.addWidget(ax1)

        # Generate plot lines
        l1 = pg.ErrorBarItem()
        l2 = pg.ScatterPlotItem(size=10, symbol='x',
                                pen=pg.mkPen(color=self.PLOTCOLORS[0]))
        l3 = ax1.plot(pen=pg.mkPen(color=self.PLOTCOLORS[1], width=1.0))
        ax1.addItem(l1)
        ax1.addItem(l2)

        # Store graph objects
        self.plot_axes['main'] = [ax0, ax1]
        self.plot_lines['main'] = [l0, l1, l2, l3]

    # =========================================================================
    # Add fit window
    # =========================================================================

    def generateNewWindow(self):
        """Get a new window name from a popup and add it."""
        # Run new window wizard
        dialog = NewWindowWizard(self)
        if not dialog.exec():
            return
        self.addFitWindow(**dialog.info)

    def addFitWindow(self, name):
        """Add a new analysis fit window."""
        # Check if the name exists
        if name in self.windows:
            logger.warning(f'{name} window already exists!')
            return

        # Generate the new tabs
        self.inputTabs[name] = QWidget()
        self.inputTabHolder.addTab(self.inputTabs[name], name)
        self.outputTabs[name] = QTabWidget()
        self.outputTabHolder.addTab(self.outputTabs[name], name)

        # Create widget holder
        winWidgets = Widgets()

        # Inputs tab ==========================================================

        # Generate the layout
        layout = QGridLayout(self.inputTabs[name])
        layout.setAlignment(Qt.AlignTop)

        # Create inputs for the fit window
        layout.addWidget(QRightLabel('Start Wavenumber (cm-1)'), 0, 0)
        winWidgets['wn_start'] = SpinBox(0, [0, 1e5])
        layout.addWidget(winWidgets['wn_start'], 0, 1)
        layout.addWidget(QRightLabel('Stop Wavenumber (cm-1)'), 1, 0)
        winWidgets['wn_stop'] = SpinBox(0, [0, 1e5])
        layout.addWidget(winWidgets['wn_stop'], 1, 1)

        # Create a checkbox to disable/enable the window
        winWidgets['run_window'] = QCheckBox('Run\nWindow?')
        winWidgets['run_window'].setChecked(True)
        layout.addWidget(winWidgets['run_window'], 0, 4)
        layout.addWidget(winWidgets['run_window'], 0, 2)

        # Create a button to remove the window
        btn = QPushButton('Remove')
        btn.clicked.connect(lambda: self.remFitWindow(name))
        layout.addWidget(btn, 1, 2)

        paramTabHolder = QTabWidget()
        layout.addWidget(paramTabHolder, 2, 0, 1, 4)
        gasTab = QScrollArea()
        paramTab = QScrollArea()
        paramTabHolder.addTab(gasTab, 'Gas Parameters')
        paramTabHolder.addTab(paramTab, 'Other Parameters')

        # Add parameter tables
        winWidgets['gasTable'] = paramTable(gasTab, 'param', width=420,
                                            gas_list=self.gas_list.keys())

        # Link the parameter table to the plot parameter combobox
        winWidgets['gasTable'].cellChanged.connect(
            lambda: self.update_plot_species(name)
        )

        playout = QGridLayout(paramTab)
        playout.setAlignment(Qt.AlignTop)

        # Background apriori
        playout.addWidget(QLabel('Apriori\nBackground'), 0, 0)
        winWidgets['bg_poly_apriori'] = DSpinBox(0, [0, 1e30])
        playout.addWidget(winWidgets['bg_poly_apriori'], 0, 1)

        # Background n params
        playout.addWidget(QLabel('Num. Background\nParams'), 0, 3)
        winWidgets['n_bg_poly'] = SpinBox(1, [1, 100])
        playout.addWidget(winWidgets['n_bg_poly'], 0, 4)

        # Shift n params and apriori
        playout.addWidget(QLabel('Apriori\nShift'), 1, 0)
        winWidgets['shift_apriori'] = DSpinBox(0, [-1000, 1000])
        playout.addWidget(winWidgets['shift_apriori'], 1, 1)
        winWidgets['fit_shift'] = QCheckBox('Fit?')
        playout.addWidget(winWidgets['fit_shift'], 1, 2)
        winWidgets['fit_shift'].setChecked(True)
        playout.addWidget(QLabel('Num. Shift\nParams'), 1, 3)
        winWidgets['n_shift'] = SpinBox(0, [0, 100])
        playout.addWidget(winWidgets['n_shift'], 1, 4)

        # Offset n params and apriori
        playout.addWidget(QLabel('Apriori\nOffset'), 2, 0)
        winWidgets['offset_apriori'] = DSpinBox(0, [-1000, 1000])
        playout.addWidget(winWidgets['offset_apriori'], 2, 1)
        winWidgets['fit_offset'] = QCheckBox('Fit?')
        playout.addWidget(winWidgets['fit_offset'], 2, 2)
        winWidgets['fit_offset'].setChecked(True)
        playout.addWidget(QLabel('Num. Offset\nParams'), 2, 3)
        winWidgets['n_offset'] = SpinBox(0, [0, 100])
        playout.addWidget(winWidgets['n_offset'], 2, 4)

        # Outputs tab =========================================================

        # Make graph and table tabs
        graphTab = QWidget()
        tableTab = QWidget()
        self.outputTabs[name].addTab(graphTab, 'Graphs')
        self.outputTabs[name].addTab(tableTab, 'Results')

        # Output graphs =======================================================

        # Setup layout
        layout = QGridLayout(graphTab)
        layout.setAlignment(Qt.AlignTop)

        # Set control for target species
        layout.addWidget(QRightLabel('Target species:'), 0, 0)
        winWidgets['target_species'] = QComboBox()
        winWidgets['target_species'].addItems([''])
        layout.addWidget(winWidgets['target_species'], 0, 1)
        winWidgets['target_species'].currentTextChanged.connect(
            lambda: self.update_window_results(name)
        )

        # Set control for plot options
        winWidgets['plot_i0'] = QCheckBox('Show I0?\n(Green)')
        layout.addWidget(winWidgets['plot_i0'], 0, 2)
        winWidgets['plot_bg'] = QCheckBox('Show Background?\n(Purple)')
        layout.addWidget(winWidgets['plot_bg'], 0, 3)
        winWidgets['plot_os'] = QCheckBox('Show Offset?\n(Red)')
        layout.addWidget(winWidgets['plot_os'], 0, 4)

        for cb in [winWidgets[k]for k in ['plot_i0', 'plot_bg', 'plot_os']]:
            cb.stateChanged.connect(lambda: self.update_window_results(name))

        # Generate the graph window
        area = da.DockArea()
        layout.addWidget(area, 1, 0, 1, 6)
        layout.addWidget(QLabel('Wavenumber (cm-1)'), 2, 2)

        # Generate the docks
        d0 = da.Dock('')
        d1 = da.Dock('')
        d2 = da.Dock('')
        area.addDock(d0, 'bottom')
        area.addDock(d1, 'bottom')
        area.addDock(d2, 'bottom')

        # Hide the titlebars
        for d in [d0, d1, d2]:
            d.hideTitleBar()

        # Generate the plot axes
        ax0 = pg.PlotWidget()
        ax1 = pg.PlotWidget()
        ax2 = pg.PlotWidget()
        ax1.setXLink(ax0)
        ax2.setXLink(ax0)

        # Add plot labels
        ax0.setLabel('left', 'Intensity (counts)')
        ax1.setLabel('left', 'Residual (%)')
        ax2.setLabel('left', 'Optical Depth')

        # Add to docks
        d0.addWidget(ax0)
        d1.addWidget(ax1)
        d2.addWidget(ax2)

        # Greate the plot lines
        pen0 = pg.mkPen(color=self.PLOTCOLORS[0], width=1.0)
        pen1 = pg.mkPen(color=self.PLOTCOLORS[1], width=1.0)
        pen2 = pg.mkPen(color=self.PLOTCOLORS[2], width=1.0, style=Qt.DashLine)
        pen3 = pg.mkPen(color=self.PLOTCOLORS[3], width=1.0, style=Qt.DashLine)
        pen4 = pg.mkPen(color=self.PLOTCOLORS[4], width=1.0, style=Qt.DashLine)
        l0 = ax0.plot(pen=pen0)  # Measured spectrum
        l1 = ax0.plot(pen=pen1)  # Fitted spectrum
        l2 = ax0.plot(pen=pen2)  # I0
        l3 = ax0.plot(pen=pen4)  # Background poly
        l4 = ax0.plot(pen=pen3)  # Offset
        l5 = ax1.plot(pen=pen0)  # residual
        l6 = ax2.plot(pen=pen0)  # Measured OD
        l7 = ax2.plot(pen=pen1)  # Fitted OD

        # Add legend to first axis
        legend = ax0.addLegend()
        legend.addItem(l0, 'Spectrum')
        legend.addItem(l1, 'Fit')

        self.plot_axes[name] = [ax0, ax1, ax2]
        self.plot_lines[name] = [l0, l1, l2, l3, l4, l5, l6, l7]

        # Add fit regions to main plot and connect to the wavenumber bounds
        self.plot_regions[name] = pg.LinearRegionItem([0, 0])
        self.plot_regions[name].setMovable(False)
        self.plot_regions[name].setToolTip(name)
        winWidgets['wn_start'].valueChanged.connect(
            lambda: self.plot_regions[name].setRegion(
                [winWidgets.get('wn_start'), winWidgets.get('wn_stop')]
            )
        )
        winWidgets['wn_stop'].valueChanged.connect(
            lambda: self.plot_regions[name].setRegion(
                [winWidgets.get('wn_start'), winWidgets.get('wn_stop')]
            )
        )
        self.plot_axes['main'][0].addItem(self.plot_regions[name])

        # Output table ========================================================

        # Generate results table
        tlayout = QGridLayout(tableTab)
        resTable = QTableWidget(0, 4)
        resTable.setHorizontalHeaderLabels(
            ['Parameter', 'Vary?', 'Fit Value', 'Fit Error'])
        resTable.horizontalHeader().setStretchLastSection(True)
        resTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        tlayout.addWidget(resTable, 0, 0)

        # Add to overall widgets ==============================================

        self.windowWidgets[name] = winWidgets
        self.results_tables[name] = resTable

        self.windows.append(name)

        # Update ratio boxes ==================================================

        # Get the current state
        xwin = self.widgets.get('ratio_WindowX')
        ywin = self.widgets.get('ratio_WindowY')

        # Clear the box
        self.widgets['ratio_WindowX'].clear()
        self.widgets['ratio_WindowY'].clear()

        # Update the list
        self.widgets['ratio_WindowX'].addItems(self.windows)
        self.widgets['ratio_WindowY'].addItems(self.windows)

        # Reset to origional state
        self.widgets.set('ratio_WindowX', xwin)
        self.widgets.set('ratio_WindowY', ywin)

        logger.info(f'{name} window added')

    def remFitWindow(self, name):
        """Remove fit window."""
        # Get the index of the window tab
        window_idx = list(self.inputTabs.keys()).index(name) + 1

        # Remove the window tab from the GUI
        self.inputTabHolder.removeTab(window_idx)
        self.outputTabHolder.removeTab(window_idx + 1)

        # Delete the actual widget from memory
        self.inputTabs[name].setParent(None)
        self.outputTabs[name].setParent(None)

        # Remove window from main plot
        self.plot_axes['main'][0].removeItem(self.plot_regions[name])

        # Remove from list of windows
        self.windows.remove(name)
        self.inputTabs.pop(name)
        self.outputTabs.pop(name)
        self.windowWidgets.pop(name)
        self.plot_lines.pop(name)
        self.plot_axes.pop(name)
        self.plot_regions.pop(name)

        # Update ratio boxes ==================================================

        # Get the current state
        xwin = self.widgets.get('ratio_WindowX')
        ywin = self.widgets.get('ratio_WindowY')

        # Clear the box
        self.widgets['ratio_WindowX'].clear()
        self.widgets['ratio_WindowY'].clear()

        # Update the list
        self.widgets['ratio_WindowX'].addItems(self.windows)
        self.widgets['ratio_WindowY'].addItems(self.windows)

        # Reset to origional state
        self.widgets.set('ratio_WindowX', xwin)
        self.widgets.set('ratio_WindowY', ywin)

        logger.info(f'{name} window removed')

    def update_plot_species(self, name):
        """Update the plot parameter options."""
        rows = self.windowWidgets[name]['gasTable'].getData()
        species_list = [r[0] for r in rows]
        current_species = self.windowWidgets[name].get('target_species')
        self.windowWidgets[name]['target_species'].clear()
        self.windowWidgets[name]['target_species'].addItems(species_list)
        self.windowWidgets[name].set('target_species', current_species)
        self.species_list[name] = species_list

    def update_ratio_combo(self, axis):
        """Update ratio selection comboboxes."""
        try:
            if axis == 'x':
                name = self.widgets.get('ratio_WindowX')
                if name != '':
                    orig_state_x = self.widgets.get('ratioSpeciesX')
                    self.widgets['ratioSpeciesX'].clear()
                    self.widgets['ratioSpeciesX'].addItems(
                        self.species_list[name])
                    self.widgets.set('ratioSpeciesX', orig_state_x)

            elif axis == 'y':
                name = self.widgets.get('ratio_WindowY')
                if name != '':
                    orig_state_x = self.widgets.get('ratioSpeciesY')
                    self.widgets['ratioSpeciesY'].clear()
                    self.widgets['ratioSpeciesY'].addItems(
                        self.species_list[name])
                    self.widgets.set('ratioSpeciesY', orig_state_x)
        except KeyError:
            pass

    # =========================================================================
    # Analysis loop and slots
    # =========================================================================

    def control_loop(self):
        """Run fit window initialisation and launch analysis."""
        # Create dictionary to hold analysers
        self.analysers = {}

        # Create dicts to hold the workers and read flags for each window
        self.analysisWorkers = {}
        self.ready_flags = {}

        # Create flag to signal correct initialisation
        self.initialisation_error_flag = False

        # Pull the widget data
        widgetData = self.getWidgetData()

        self.update_status('Initialising')

        # Ensure the output directory exists
        if not os.path.isdir(widgetData['save_dir']):
            os.makedirs(widgetData['save_dir'])

        # Generate a log file handler for this analysis loop
        self.analysis_logger = logging.FileHandler(
            f'{widgetData["save_dir"]}/ftpyr_analysis.log',
            mode='w'
        )
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_fmt = '%Y-%m-%d %H:%M:%S'
        f_format = logging.Formatter(log_fmt, date_fmt)
        self.analysis_logger.setFormatter(f_format)
        self.analysis_logger.setLevel(logging.DEBUG)
        logger.addHandler(self.analysis_logger)

        # Form the output file name
        self.outfname = f"{widgetData['save_dir']}/all_gas_output.csv"

        # Generate the setup worker
        setupWorker = SetupWorker(self.windows, widgetData, self.outfname)
        setupWorker.signals.error.connect(self.update_error)
        setupWorker.signals.initialize.connect(self.initialize_window)
        setupWorker.signals.finished.connect(self.begin_analysis)
        self.threadpool.start(setupWorker)

    def initialize_window(self, name, analyser):
        """Initialize window analyser and results table."""
        # Add the analyser to the dictionary
        self.analysers[name] = analyser

        # Clear all current table rows
        self.results_tables[name].clearContents()

        # Make the rows
        self.results_tables[name].setRowCount(len(analyser.params.keys()))

        for i, [pname, param] in enumerate(analyser.params.items()):
            self.results_tables[name].setItem(i, 0, QTableWidgetItem(pname))
            self.results_tables[name].setItem(
                i, 1, QTableWidgetItem(str(param.vary)))

    def begin_analysis(self):
        """Run main analysis loop."""
        # Check initialisation went ok
        if self.initialisation_error_flag:
            logger.info('Error with window initialisation')
            return
        logger.info('All windows initialised, begining analysis loop')

        # Pull the widget data
        widgetData = self.getWidgetData()

        # Get the spectra to analyse
        self.spectra_list = widgetData['spec_fnames'].split('\n')
        self.spectrum_counter = 0

        # Create dictionary to hold fit results
        self.fit_results = {}

        # Get the output ppmm flag and disable the option
        self.output_ppmm_flag = widgetData['output_units'] == 'ppm.m'

        self.update_status('Analysing')

        # Generate the thread workers
        for name in self.windows:

            # Generate the worker
            analysisWorker = AnalysisWorker(name, self.analysers[name])
            analysisWorker.signals.results.connect(self.get_results)
            analysisWorker.signals.error.connect(self.update_error)
            self.analysisWorkers[name] = analysisWorker
            self.ready_flags[name] = False
            self.threadpool.start(analysisWorker)

        # Send the first spectrum
        self.set_next_spectrum()

        # Disable the start button and enable the pause/stop buttons
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def send_spectrum(self, spectrum):
        """Send spectrum to workers to analyse."""
        for name, worker in self.analysisWorkers.items():
            worker.spectrum = spectrum
            self.ready_flags[name] = False

    def stop_analysis(self):
        """Stop analysis."""
        for name, worker in self.analysisWorkers.items():
            worker.stop()
        logger.debug('Analysis stopped')
        self.analysis_complete()

    def pause_analysis(self):
        """Pause the workers."""
        # Pause each worker
        for name, worker in self.analysisWorkers.items():
            worker.pause()

        # Update button label
        if self.pause_btn.text() == 'Pause':
            logger.debug('Analysis continued')
            self.pause_btn.setText('Continue')
        else:
            logger.debug('Analysis paused')
            self.pause_btn.setText('Pause')

    @Slot(tuple)
    def get_results(self, results):
        """Catch results and flag that worker is ready for next spectrum."""
        # Unpack the results
        name, fit = results

        # Add to the dictionary
        self.fit_results[name] = fit

        if fit is not None:
            self.update_window_results(name)

        # Update worker ready flag
        self.ready_flags[name] = True

        # Check if all workers are ready
        ready_flags = np.array([b for b in self.ready_flags.values()])
        if ready_flags.all():

            # Write the results
            with open(self.outfname, 'a') as outfile:

                # Write the filename and timestamp
                ts = self.spectrum.attrs['timestamp']
                outfile.write(f'{self.spec_filename},{ts}')

                # Write the gas parameter results
                for name in self.ready_flags.keys():
                    fit = self.fit_results[name]

                    for par in fit.params.values():
                        if par.species is not None:
                            outfile.write(f',{par.fit_val},{par.fit_err}')

                    # Write the fit quality result
                    outfile.write(
                        f',{fit.nerr},{fit.max_residual},{fit.std_residual}'
                    )
                outfile.write('\n')

            # If so, check if all spectra have been analysed
            if self.spectrum_counter == len(self.spectra_list):
                self.stop_analysis()

            # If not, analyse the next spectrum
            else:
                self.set_next_spectrum()

    def set_next_spectrum(self):
        """Set next spectrum to analyse."""
        # Construct the spectrum file name
        self.spec_filename = self.spectra_list[self.spectrum_counter]
        logger.debug(f'Sending {self.spec_filename} for analysis')

        # Read in the spectrum and send to the workers
        self.spectrum = read_spectrum(self.spec_filename)
        self.send_spectrum(self.spectrum)

        # Update the progress
        self.progress.setValue(
            (self.spectrum_counter+1) / len(self.spectra_list) * 100
        )
        self.spectrum_counter += 1

        # Update the results plots
        self.update_main_plots()
        self.update_ratio_plot()

    def update_main_plots(self):
        """Update the main spectrum graph."""
        # Get the spectrum x and y data
        x = self.spectrum.coords['Wavenumber'].to_numpy()
        y = self.spectrum.to_numpy()
        self.plot_lines['main'][0].setData(x, y)

    def update_window_results(self, name):
        """Update the window results graphs and table."""
        try:
            # Pull the window results
            fit = self.fit_results[name]
        except KeyError:
            return

        # Get the plot parameter
        plot_gas = self.windowWidgets[name].get('target_species')

        # Update the fit plot
        self.plot_lines[name][0].setData(fit.grid, fit.spec)
        self.plot_lines[name][1].setData(fit.grid, fit.fit)

        # Add optional lines
        if self.windowWidgets[name].get('plot_i0'):
            self.plot_lines[name][2].setData(
                fit.grid, fit.bg_poly + np.nan_to_num(fit.offset)
            )
        else:
            self.plot_lines[name][2].setData([], [])
        if self.windowWidgets[name].get('plot_bg'):
            self.plot_lines[name][3].setData(fit.grid, fit.bg_poly)
        else:
            self.plot_lines[name][3].setData([], [])
        if self.windowWidgets[name].get('plot_os'):
            self.plot_lines[name][4].setData(fit.grid, fit.offset)
        else:
            self.plot_lines[name][4].setData([], [])

        # Add residual
        self.plot_lines[name][5].setData(fit.grid, fit.residual)

        # Add optical depths
        try:
            self.plot_lines[name][6].setData(fit.grid, fit.meas_od[plot_gas])
            self.plot_lines[name][7].setData(fit.grid, fit.fit_od[plot_gas])
        except KeyError:
            pass

        # Update results table
        for i, p in enumerate(fit.params.values()):

            # Check if gases require conversion to ppm.m
            if p.species is not None and self.output_ppmm_flag:
                val = p.fit_val_to_ppmm()
                err = p.fit_err_to_ppmm()
            else:
                val = p.fit_val
                err = p.fit_err

            # Update the table
            self.results_tables[name].setItem(i, 2, QTableWidgetItem(str(val)))
            self.results_tables[name].setItem(i, 3, QTableWidgetItem(str(err)))

    def update_ratio_plot(self):
        """Update the data shown on the ratio plot."""
        # Read in the time series results for the ratio plots
        try:
            df = pd.read_csv(self.outfname, parse_dates=['Timestamp'])
        except AttributeError:
            return
        xwin = self.widgets.get('ratio_WindowX')
        xgas = self.widgets.get('ratioSpeciesX')
        ywin = self.widgets.get('ratio_WindowY')
        ygas = self.widgets.get('ratioSpeciesY')

        try:
            # Remove bad fits if desired
            if self.widgets.get('bad_fit_flag'):
                idx = np.logical_and(
                    df[f'FitQuality ({xwin})'] == 0,
                    df[f'FitQuality ({ywin})'] == 0
                )
                df = df[idx]

            # Unpack good fit values and errors
            xval = df[f'{xgas} ({xwin})'].to_numpy()
            xerr = df[f'{xgas}_err ({xwin})'].to_numpy()
            yval = df[f'{ygas} ({ywin})'].to_numpy()
            yerr = df[f'{ygas}_err ({ywin})'].to_numpy()

            # Update plot
            self.plot_lines['main'][1].setData(
                x=xval, y=yval, height=yerr, width=xerr, beam=10,
                pen=pg.mkPen(color=self.PLOTCOLORS[0], width=0.0))
            self.plot_lines['main'][2].setData(x=xval, y=yval)

            # Fit linear regression if there is more than 1 point
            if len(xval) > 1:
                xfit = np.array([xval.min(), xval.max()])

                # Run initial polyfit to get first guess of parameters
                [m, c] = np.polyfit(xval, yval, 1)

                # Fit without taking errors into account
                if not self.widgets.get('ratio_fit_errors'):
                    popt, pcov = curve_fit(
                        lin_fit,
                        xval,
                        yval,
                        [m, c]
                    )
                    perr = np.sqrt(np.diag(pcov))

                # Fit with taking errors into account
                else:
                    data = odr.Data(
                        x=xval,
                        y=yval,
                        wd=np.power(xerr, -1),
                        we=np.power(yerr, -1)
                    )

                    myodr = odr.ODR(
                        data,
                        odr.unilinear,
                        beta0=[m, c])
                    out = myodr.run()
                    popt = out.beta
                    perr = out.sd_beta

                # Make the best fit line
                yfit = lin_fit(xfit, *popt)

                # pdate the plots
                self.plot_lines['main'][3].setData(xfit, yfit)
                self.ratio_gradient.setText(
                    f'{popt[0]:.2E}\n(+/- {perr[0]:.2E})')
                self.ratio_intercept.setText(
                    f'{popt[1]:.2E}\n(+/- {perr[1]:.2E})')

            # Otherwise clear the graphs
            else:
                self.plot_lines['main'][3].setData([], [])
                self.ratio_gradient.setText('-')
                self.ratio_intercept.setText('-')
        except KeyError:
            pass

    def plot_first_spectrum(self):
        """Plot first spectrum in list."""
        try:
            filename = self.widgets.get('spec_fnames').split('\n')[0]
            if filename != self.first_filename:
                self.first_filename = filename
                spectrum = read_spectrum(filename)
                x = spectrum.coords['Wavenumber'].to_numpy()
                y = spectrum.to_numpy()
                self.plot_lines['main'][0].setData(x, y)

        except Exception:
            pass

    def update_error(self, error):
        """Update error messages from the worker."""
        exctype, value, trace = error
        logger.error(f'Uncaught exception!\n{trace}')
        self.initialisation_error_flag = True
        self.stop_analysis()

    def update_status(self, status):
        """Update status bar."""
        self.statusBar().showMessage(status)

    def analysis_complete(self):
        """Signal for end of analysis."""
        # Renable the start button
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText('Pause')

        # Set the status bar
        self.update_status('Ready')

        # Log end of analysis
        logger.info('Analysis finished')

        # Turn off the logger
        logger.removeHandler(self.analysis_logger)

    # =========================================================================
    # Program Global Slots
    # =========================================================================

    @Slot(str, logging.LogRecord)
    def updateLog(self, status, record):
        """Write log statements to the logBox widget."""
        color = self.LOGCOLORS.get(record.levelno, 'black')
        s = '<pre><font color="%s">%s</font></pre>' % (color, status)
        self.logBox.appendHtml(s)

    @Slot(QWidget, str, list)
    def browse(self, widget, mode='single', filter=None):
        """Open native file dialogue."""
        # Check if specified file extensions
        if filter is not None:
            filter = filter + ';;All Files (*)'

        # Pick a single file to read
        if mode == 'single':
            fname, _ = QFileDialog.getOpenFileName(
                self, 'Select File', '', filter)

        elif mode == 'multi':
            fname, _ = QFileDialog.getOpenFileNames(self, 'Select Files', '',
                                                    filter)

        elif mode == 'save':
            fname, _ = QFileDialog.getSaveFileName(self, 'Save As', '', filter)

        elif mode == 'folder':
            fname = QFileDialog.getExistingDirectory(self, 'Select Folder')

        # Get current working directory
        cwd = os.getcwd() + '/'
        cwd = cwd.replace("\\", "/")

        # Update the relavant widget for a single file
        if type(fname) == str and fname != '':
            # if cwd in fname:
            #     fname = fname[len(cwd):]
            widget.setText(fname)

        # And for multiple files
        elif type(fname) == list and fname != []:
            for i, f in enumerate(fname):
                if cwd in f:
                    fname[i] = f[len(cwd):]
            widget.setText('\n'.join(fname))

    # =========================================================================
    # Program settings and theme
    # =========================================================================

    def getWidgetData(self):
        """Get the widget data into a single dictionary."""
        widgetData = {}

        # Save the main gui widgets
        for label in self.widgets:
            widgetData[label] = self.widgets.get(label)

        # Save the window widgets
        widgetData['fitWindows'] = {}
        for name in self.windowWidgets:
            winConfig = {}

            for key in self.windowWidgets[name]:
                if key == 'gasTable':
                    winConfig[key] = self.windowWidgets[name][key].getData()
                else:
                    winConfig[key] = self.windowWidgets[name].get(key)

            widgetData['fitWindows'][name] = winConfig

        return widgetData

    def saveConfig(self, asksavepath=True):
        """Save the program configuration."""
        # Pull the main widget data
        config = self.getWidgetData()

        # Save the theme
        config['theme'] = self.theme

        # Save the main gui widgets
        for label in self.widgets:
            config[label] = self.widgets.get(label)

        # Save the window widgets
        config['fitWindows'] = {}
        for name in self.windowWidgets:
            winConfig = {}

            for key in self.windowWidgets[name]:
                if key == 'gasTable':
                    winConfig[key] = self.windowWidgets[name][key].getData()
                else:
                    winConfig[key] = self.windowWidgets[name].get(key)

            config['fitWindows'][name] = winConfig

        # Get save filename if required
        if asksavepath or self.config_fname is None:
            filter = 'YAML (*.yml *.yaml);;All Files (*)'
            fname, s = QFileDialog.getSaveFileName(self, 'Save Config', '',
                                                   filter)
            # If valid, proceed. If not, return
            if fname != '' and fname is not None:
                self.config_fname = fname
            else:
                return

        # Write the config
        with open(self.config_fname, 'w') as outfile:
            yaml.dump(config, outfile)

        # Log the update
        logger.info(f'Config file saved to {self.config_fname}')

        # Record the default settings path
        with open('bin/.config', 'w') as w:
            w.write(self.config_fname)

        self.config = config

    def loadConfig(self, fname=None):
        """Read the config file."""
        if fname is None:
            filter = 'YAML (*.yml *.yaml);;All Files (*)'
            fname, tfile = QFileDialog.getOpenFileName(
                self, 'Load Config', '', filter
            )

        # Open the config file
        try:
            with open(fname, 'r') as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.FullLoader)

            logger.info(f'Loading config from {self.config_fname}')

            # Clear current windows
            for name in list(self.windows):
                self.remFitWindow(name)

            # Apply each config setting
            for label, value in config.items():

                # Set the fit windows
                if label == 'fitWindows':
                    for name, widgets in value.items():

                        # Generate the window tabs
                        self.addFitWindow(name)

                        for key, val in widgets.items():

                            # Setup the gas parameter table
                            if key == 'gasTable':
                                self.windowWidgets[name][key].setData(val)

                            # Set other widgets
                            else:
                                self.windowWidgets[name].set(key, val)

                elif label == 'theme':
                    self.theme = value

                else:
                    self.widgets.set(label, value)

            # Update the config file settings
            self.config_fname = fname
            with open('bin/.config', 'w') as w:
                w.write(self.config_fname)

        except FileNotFoundError:
            logger.warning(f'Unable to load config file {self.config_fname}')
            config = {}
        self.config = config
        return config

    def changeTheme(self):
        """Change the theme between light and dark."""
        if self.theme == 'Light':
            # Set overall style
            self.app.setStyleSheet(qdarktheme.load_stylesheet())
            bg_color = 'k'
            plotpen = pg.mkPen('darkgrey', width=1)
            self.theme = 'Dark'
        elif self.theme == 'Dark':
            # Set overall style
            self.app.setStyleSheet(qdarktheme.load_stylesheet("light"))
            bg_color = 'w'
            plotpen = pg.mkPen('k', width=1)
            self.theme = 'Light'

        # Set axes spines color
        for axes in self.plot_axes.values():
            for ax in axes:
                ax.setBackground(bg_color)
                ax.getAxis('left').setPen(plotpen)
                ax.getAxis('right').setPen(plotpen)
                ax.getAxis('top').setPen(plotpen)
                ax.getAxis('bottom').setPen(plotpen)
                ax.getAxis('left').setTextPen(plotpen)
                ax.getAxis('bottom').setTextPen(plotpen)


# =============================================================================
# =============================================================================
# Worker threads
# =============================================================================
# =============================================================================

class WorkerSignals(QObject):
    """Signals for Worker classes to communicate with the main thread."""
    results = Signal(tuple)
    finished = Signal()
    error = Signal(tuple)
    initialize = Signal(str, object)


class SetupWorker(QRunnable):
    """Worker class to handle analyser setup in a seperate thread."""

    def __init__(self, windows, widgetData, outfname, *args, **kwargs):
        """Initialise."""
        super(SetupWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.windows = windows
        self.widgetData = widgetData
        self.outfname = outfname

    @Slot()
    def run(self):
        """Worker run function."""
        try:
            self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        self.signals.finished.emit()

    def fn(self):
        """Main setup loop."""
        # Open main output file
        with open(self.outfname, 'w') as main_outfile:

            # Start header line
            main_outfile.write('Filename,Timestamp')

            # Generate the initialisation workers
            for name in self.windows:

                windowData = self.widgetData['fitWindows'][name]

                if not windowData['run_window']:
                    continue

                # Setup the parameters
                params = Parameters()

                # Add gas parameters
                for line in windowData['gasTable']:
                    params.add(
                        name=line[0],
                        vary=line[1],
                        species=line[2],
                        temp=line[3],
                        pres=line[4],
                        path=line[5]
                    )

                    # Add outfile header
                    main_outfile.write(
                        f',{line[0]} ({name}),{line[0]}_err ({name})'
                    )

                main_outfile.write(
                    f',FitQuality ({name}),MaxResidual ({name}),'
                    f'StdevResidual ({name})'
                )

                # Add background parameters
                for i in range(windowData['n_bg_poly']):
                    if i == 0:
                        value = windowData['bg_poly_apriori']
                    else:
                        value = 0
                    params.add(name=f'bg_poly{i}', value=value)

                # Add shift parameters
                for i in range(windowData['n_shift']):
                    if i == 0:
                        value = windowData['shift_apriori']
                    else:
                        value = 0
                    params.add(name=f'shift{i}', value=value,
                               vary=windowData['fit_shift'])

                # Add offset parameters
                for i in range(windowData['n_offset']):
                    if i == 0:
                        value = windowData['offset_apriori']
                    else:
                        value = 0
                    params.add(name=f'offset{i}', value=value,
                               vary=windowData['fit_offset'])

                # Add ILS parameters
                params.add(
                    name='fov',
                    value=self.widgetData['fov'],
                    vary=self.widgetData['fit_fov']
                )
                params.add(
                    name='opd',
                    value=self.widgetData['opd'],
                    vary=False
                )

                # Setup analyser settings
                logger.info(f'Generating analyser for {name} window')
                outfile = f"{self.widgetData['save_dir']}/{name}_output.csv"
                output_ppmm_flag = self.widgetData['output_units'] == 'ppm.m'
                analyser_settings = {
                    'params': params,
                    'rfm_path': self.widgetData['rfm_path'],
                    'hitran_path': self.widgetData['hitran_path'],
                    'wn_start': windowData['wn_start'],
                    'wn_stop': windowData['wn_stop'],
                    'zero_fill_factor': self.widgetData['zero_fill_factor'],
                    'solar_flag': self.widgetData['solar_flag'],
                    'obs_height': self.widgetData['obs_height'],
                    'update_params': self.widgetData['update_params'],
                    'residual_limit': self.widgetData['residual_limit'],
                    'npts_per_cm': 100,
                    'apod_function': self.widgetData['apod_function'],
                    'outfile': outfile,
                    'output_ppmm_flag': output_ppmm_flag
                }

                # Log analyser settings
                logger.debug('Analyser settings:')
                for key, value in analyser_settings.items():
                    logger.debug(f'{key}: {value}')

                # Generate the analyser function
                self.analyser = Analyser(**analyser_settings)

                # Setup the window in the front end
                self.signals.initialize.emit(name, self.analyser)

            # Start a new line from the header
            main_outfile.write('\n')


class AnalysisWorker(QRunnable):
    """Worker class to handle spectra analysis in a separate thread."""

    def __init__(self, name, analyser, *args, **kwargs):
        """Initialise."""
        super(AnalysisWorker, self).__init__()
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.isStopped = False
        self.isPaused = False
        self.spectrum = None
        self.analyser = analyser
        self.initialized = False

    @Slot()
    def run(self):
        """Worker run function."""
        try:
            self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.initialized = True
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        logger.info(f'{self.name} analyser finished')

    def fn(self):
        """Main analysis loop."""
        while not self.isStopped:
            if self.spectrum is not None and not self.isPaused:
                fit = self.analyser.fit(self.spectrum)
                self.spectrum = None
                self.signals.results.emit([self.name, fit])
            else:
                time.sleep(0.001)

    def pause(self):
        """Pause the analysis."""
        if self.isPaused:
            logger.debug(f'{self.name} analyser played')
            self.isPaused = False
        else:
            logger.debug(f'{self.name} analyser paused')
            self.isPaused = True

    def stop(self):
        """Stop the analysis."""
        logger.debug(f'{self.name} analyser stopped')
        self.isStopped = True


# =============================================================================
# =============================================================================
# Coustom Widgets
# =============================================================================
# =============================================================================

# =============================================================================
# New fit window wizard
# =============================================================================

class NewWindowWizard(QDialog):
    """Opens a wizard to define a new station."""

    def __init__(self, parent=None):
        """Initialise the window."""
        super(NewWindowWizard, self).__init__(parent)

        # Set the window properties
        self.setWindowTitle('Add new fit window')

        self._createApp()

    def _createApp(self):
        # Set the layout
        layout = QGridLayout()
        self.wname = QLineEdit()
        layout.addWidget(QLabel('Name:'), 0, 0)
        layout.addWidget(self.wname, 0, 1)

        # Add cancel and accept buttons
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.cancel_action)
        accept_btn = QPushButton('Accept')
        accept_btn.clicked.connect(self.accept_action)
        layout.addWidget(accept_btn, 3, 0)
        layout.addWidget(cancel_btn, 3, 1)

        self.setLayout(layout)

    def accept_action(self):
        """Record the window data and exit."""
        self.info = {
            'name': str(self.wname.text())}
        self.accept()

    def cancel_action(self):
        """Close the window without creating a new station."""
        self.info = {}
        self.close()


# =============================================================================
# Widget Holder
# =============================================================================

class Widgets(dict):
    """Object to allow easy config/info transfer with Qt Widgets."""

    def __init__(self):
        """Initialise."""
        super().__init__()

    def get(self, key):
        """Get the value of a widget."""
        if key not in self.keys():
            logger.warning(f'{key} widget not found!')
            return
        if type(self[key]) == QTextEdit:
            return self[key].toPlainText()
        elif type(self[key]) == QLineEdit:
            return self[key].text()
        elif type(self[key]) == QComboBox:
            return str(self[key].currentText())
        elif type(self[key]) == QCheckBox:
            return self[key].isChecked()
        elif type(self[key]) in [QSpinBox, QDoubleSpinBox, SpinBox, DSpinBox]:
            return self[key].value()
        else:
            raise ValueError('Widget type not recognised!')
            return

    def set(self, key, value):
        """Set the value of a widget."""
        if key not in self.keys():
            logger.warning(f'{key} widget not found!')
        elif type(self[key]) in [QTextEdit, QLineEdit]:
            self[key].setText(str(value))
        elif type(self[key]) == QComboBox:
            index = self[key].findText(value, Qt.MatchFixedString)
            if index >= 0:
                self[key].setCurrentIndex(index)
        elif type(self[key]) == QCheckBox:
            self[key].setChecked(value)
        elif type(self[key]) in [QSpinBox, QDoubleSpinBox, SpinBox, DSpinBox]:
            self[key].setValue(float(value))
        else:
            raise ValueError('Widget type not recognised!')


# =============================================================================
# Parameter Table
# =============================================================================

class paramTable(QTableWidget):
    """Object to build parameter tables."""

    def __init__(self, parent, type, width=None, height=None, gas_list=None):
        """Initialise."""
        super().__init__(parent)

        self.gas_list = gas_list
        self._width = width
        self._height = height
        self._type = type

        if self._width is not None:
            self.setFixedWidth(self._width)
        if self._height is not None:
            self.setFixedHeight(self._height)

        if self._type == 'param':
            self._param_table()

        if self._type == 'poly':
            self._poly_table()

        self.itemChanged.connect(self._resize)
        self._resize()
        self.verticalHeader().setVisible(False)

    def _param_table(self):
        """Create a parameter table."""
        self.setColumnCount(6)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(
            ['Name', 'Vary?', 'Species', 'Temp(K)', 'Pres(mb)',
             'Pathlength(m)']
        )

    def _poly_table(self):
        """Create a polynomial table."""
        self.setColumnCount(2)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Value', 'Vary?'])

    def add_row(self):
        """Add row to the bottom of the table."""
        n = self.rowCount()
        self.setRowCount(n+1)

        if self._type == 'param':
            cb = QCheckBox()
            cb.setChecked(True)
            self.setCellWidget(n, 1, cb)
            dm = QComboBox()
            dm.addItems(self.gas_list)
            self.setCellWidget(n, 2, dm)
            self.setCellWidget(n, 3, DSpinBox(293, [-294, 1e6]))
            self.setCellWidget(n, 4, DSpinBox(1000, [0, 1e6]))
            self.setCellWidget(n, 5, DSpinBox(100, [0, 1e16]))

        if self._type == 'poly':
            cb = QCheckBox()
            cb.setChecked(True)
            self.setItem(n, 0, QTableWidgetItem('0.0'))
            self.setCellWidget(n, 1, cb)

        self._resize()

    def rem_row(self):
        """Remove the last row from the table."""
        rows = [i.row() for i in self.selectedIndexes()]
        for row in sorted(rows, reverse=True):
            self.removeRow(row)
        self._resize()

    def contextMenuEvent(self, event):
        """Set up right click to add/remove rows."""
        menu = QMenu(self)
        addAction = menu.addAction('Add')
        remAction = menu.addAction('Remove')
        action = menu.exec(self.mapToGlobal(event.pos()))
        if action == addAction:
            self.add_row()
        if action == remAction:
            self.rem_row()

    def setData(self, data):
        """Populate the table using saved config."""
        for i in range(len(data)):
            self.add_row()
            line = data[i]
            if self._type == 'param':
                self.setItem(i, 0, QTableWidgetItem(line[0]))
                self.cellWidget(i, 1).setChecked(line[1])
                index = self.cellWidget(i, 2).findText(
                    line[2], Qt.MatchFixedString
                )
                if index >= 0:
                    self.cellWidget(i, 2).setCurrentIndex(index)
                self.cellWidget(i, 3).setValue(line[3])
                self.cellWidget(i, 4).setValue(line[4])
                self.cellWidget(i, 5).setValue(line[5])

            elif self._type == 'poly':
                self.setItem(i, 0, QTableWidgetItem(str(line[0])))
                self.cellWidget(i, 1).setChecked(line[1])

    def getData(self):
        """Extract the information from the table."""
        # Get number of rows
        nrows = self.rowCount()
        data = []

        try:
            # Read the data from a param table
            if self._type == 'param' and nrows > 0:
                for i in range(nrows):
                    row = [
                        self.item(i, 0).text(),
                        self.cellWidget(i, 1).isChecked(),
                        self.cellWidget(i, 2).currentText(),
                        self.cellWidget(i, 3).value(),
                        self.cellWidget(i, 4).value(),
                        self.cellWidget(i, 5).value()
                    ]
                    data.append(row)

            # Read the data from a poly table
            elif self._type == 'poly' and nrows > 0:
                for i in range(nrows):

                    row = [float(self.item(i, 0).text()),
                           self.cellWidget(i, 1).isChecked()]
                    data.append(row)
        except AttributeError:
            pass

        return data

    def _resize(self):
        """Autoscale the table."""
        self.setColumnWidth(0, 60)
        self.setColumnWidth(1, 50)
        if self._type == 'param':
            self.setColumnWidth(2, 60)
            self.setColumnWidth(3, 70)
            self.setColumnWidth(4, 70)
            self.setColumnWidth(5, 100)


# =============================================================================
# Aligned labels
# =============================================================================

class QRightLabel(QLabel):
    """Right aligned QLabel"""

    def __init__(self, label):
        super().__init__(label)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)


# =============================================================================
# Spinbox classes
# =============================================================================

# Create a Spinbox object for ease
class DSpinBox(QDoubleSpinBox):
    """Object for generating custom float spinboxes."""

    def __init__(self, value, range=None, step=1.0):
        """Initialise."""
        super().__init__()
        if range is not None:
            self.setRange(*range)
        self.setValue(value)
        self.setSingleStep(step)


class SpinBox(QSpinBox):
    """Object for generating custom integer spinboxes."""

    def __init__(self, value, range):
        """Initialise."""
        super().__init__()
        self.setRange(*range)
        self.setValue(value)


# =============================================================================
# Divider classes
# =============================================================================

class QHLine(QFrame):
    """Horizontal line widget."""

    def __init__(self):
        """Initialise."""
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):
    """Vertical line widget."""

    def __init__(self):
        """Initialise."""
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


# =============================================================================
# Simple linear fit model
# =============================================================================

def lin_fit(x, m, c):
    return x * m + c

# =============================================================================
# =============================================================================
# Client code
# =============================================================================
# =============================================================================


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(app)
    window.show()
    sys.exit(app.exec())
