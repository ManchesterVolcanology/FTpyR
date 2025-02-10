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
from PySide2.QtGui import QIcon
from PySide2.QtCore import (
    Qt, QObject, QTimer, Signal, Slot, QThreadPool, QRunnable
)
from PySide2.QtWidgets import (
    QMainWindow, QScrollArea, QGridLayout, QApplication, QToolBar, QFrame,
    QSplitter, QProgressBar, QLabel, QLineEdit, QTextEdit, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QPushButton,
    QFileDialog, QWidget, QTabWidget, QDialog, QTableWidget, QTableWidgetItem,
    QMenu, QHeaderView, QAction
)
import pyqtgraph as pg
import pyqtgraph.dockarea as da

from ftpyr.read import read_spectrum
from ftpyr.analyse import Analyser
from ftpyr.parameters import Parameters, Layer


__version__ = '0.2.0'

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
        # self.setGeometry(40, 40, 1210, 700)
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

        # Add menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveAct)
        fileMenu.addAction(saveasAct)
        fileMenu.addAction(loadAct)
        toolMenu = menubar.addMenu('&View')
        toolMenu.addAction(themeAct)

        # Create a toolbar
        # toolBar = QToolBar("Main toolbar")
        # self.addToolBar(toolBar)
        # toolBar.addAction(saveAct)
        # toolBar.addAction(saveasAct)
        # toolBar.addAction(loadAct)
        # toolBar.addAction(themeAct)

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
        self.windowTabs = {}
        self.windowWidgets = {}
        self.windows = []
        self.windowTextBoxes = {}
        self.layerWidgets = {}
        self.layerTabHolders = {}
        self.plot_axes = {}
        self.plot_lines = {}
        self.cov_plot = {}
        self.cov_imview = {}
        self.plot_regions = {}
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

        settingsTabs = QTabWidget()
        layout.addWidget(settingsTabs, 0, 0)

        # File selection ======================================================

        # Add tabs
        fileTab = QWidget()
        settingsTabs.addTab(fileTab, 'File Setup')

        # Add layout
        file_layout = QGridLayout(fileTab)
        file_layout.setAlignment(Qt.AlignTop)
        nrow = 0

        # Add an input for the spectra selection
        file_layout.addWidget(QLabel('Spectra:'), nrow, 0)
        self.widgets['spec_fnames'] = QTextEdit()
        file_layout.addWidget(self.widgets['spec_fnames'], nrow, 1, 1, 3)
        self.widgets['spec_fnames'].textChanged.connect(
            self.plot_first_spectrum
        )
        self.first_filename = ''
        btn = QPushButton('Browse')
        btn.clicked.connect(
            partial(self.browse, self.widgets['spec_fnames'], 'multi', None)
        )
        file_layout.addWidget(btn, nrow, 4)
        nrow += 3

        # # Add an input for a directory to watch
        # file_layout.addWidget(QLabel('Watch Folder:'), nrow, 0)
        # self.widgets['watch_dir'] = QLineEdit()
        # file_layout.addWidget(self.widgets['watch_dir'], nrow, 1, 1, 3)
        # btn = QPushButton('Browse')
        # btn.clicked.connect(
        #     partial(self.browse, self.widgets['watch_dir'], 'folder', None)
        # )
        # file_layout.addWidget(btn, nrow, 4)
        # nrow += 1

        # # Add an input for a directory to watch
        # file_layout.addWidget(QLabel('Analysis\nMode:'), nrow, 0)
        # self.widgets['analysis_mode'] = QComboBox()
        # self.widgets['analysis_mode'].addItems(['Post-Process', 'Real-Time'])
        # self.widgets['analysis_mode'].currentTextChanged.connect(
        #     self.flip_mode
        # )
        # file_layout.addWidget(self.widgets['analysis_mode'], nrow, 1, 1, 3)
        # nrow += 1

        # Add an input for the save selection
        file_layout.addWidget(QLabel('Output\nFolder:'), nrow, 0)
        self.widgets['save_dir'] = QLineEdit('Results')
        file_layout.addWidget(self.widgets['save_dir'], nrow, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.clicked.connect(
            partial(self.browse, self.widgets['save_dir'], 'folder', None)
        )
        file_layout.addWidget(btn, nrow, 4)
        nrow += 1

        # RFM setup ===========================================================

        # Add tab
        rfmTab = QWidget()
        settingsTabs.addTab(rfmTab, 'RFM')

        # Add layout
        rfm_layout = QGridLayout(rfmTab)
        rfm_layout.setAlignment(Qt.AlignTop)

        # Add an input for the RFM exe path
        rfm_layout.addWidget(QLabel('RFM:'), 0, 0)
        self.widgets['rfm_path'] = QLineEdit()
        rfm_layout.addWidget(self.widgets['rfm_path'], 0, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.clicked.connect(
            partial(self.browse, self.widgets['rfm_path'], 'single', None)
        )
        rfm_layout.addWidget(btn, 0, 4)

        # Add an input for the HITRAN database path
        rfm_layout.addWidget(QLabel('HITRAN:'), 1, 0)
        self.widgets['hitran_path'] = QLineEdit()
        rfm_layout.addWidget(self.widgets['hitran_path'], 1, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.clicked.connect(
            partial(self.browse, self.widgets['hitran_path'], 'single', None)
        )
        rfm_layout.addWidget(btn, 1, 4)

        # Add an input for the VMR input file
        rfm_layout.addWidget(QLabel('VMR Database:'), 2, 0)
        self.widgets['vmr_file'] = QLineEdit()
        rfm_layout.addWidget(self.widgets['vmr_file'], 2, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.clicked.connect(
            partial(self.browse, self.widgets['vmr_file'], 'single', None)
        )
        rfm_layout.addWidget(btn, 2, 4)

        # Model setup =========================================================

        # Add tab
        specTab = QWidget()
        settingsTabs.addTab(specTab, 'Model')

        # Add layout
        spec_layout = QGridLayout(specTab)
        spec_layout.setAlignment(Qt.AlignTop)

        # General settings ----------------------------------------------------
        nrow = 0

        # Input for zero filling
        spec_layout.addWidget(QRightLabel('Zero Filling\nFactor'), nrow, 0)
        self.widgets['zero_fill_factor'] = SpinBox(0, [0, 100])
        spec_layout.addWidget(self.widgets['zero_fill_factor'], nrow, 1)

        spec_layout.addWidget(QRightLabel('Output Units:'), nrow, 2)
        self.widgets['output_units'] = QComboBox()
        self.widgets['output_units'].addItems(['molecules.cm-2', 'ppm.m'])
        spec_layout.addWidget(self.widgets['output_units'], nrow, 3)

        nrow += 1

        # Input for model grid spacing
        spec_layout.addWidget(QRightLabel('Model Points\nper cm'), nrow, 0)
        self.widgets['pts_per_cm'] = SpinBox(100, [0, 1e8])
        spec_layout.addWidget(self.widgets['pts_per_cm'], nrow, 1)

        # Input for model grid padding
        spec_layout.addWidget(
            QRightLabel('Model Grid\npadding (cm-1)'), nrow, 2
        )
        self.widgets['model_padding'] = SpinBox(10, [0, 1e4])
        spec_layout.addWidget(self.widgets['model_padding'], nrow, 3)

        nrow += 1

        # Solar Occultation settings ------------------------------------------
        # spec_layout.addWidget(QHLine(), nrow, 0, 1, 10)
        # nrow += 1

        # # Add control for solar occultation
        # self.widgets['solar_flag'] = QCheckBox('Solar\nOccultation')
        # spec_layout.addWidget(self.widgets['solar_flag'], nrow, 0)

        # # Add control for observation height
        # spec_layout.addWidget(QRightLabel('Observation\nHeight (m)'), nrow, 1)
        # self.widgets['obs_height'] = DSpinBox(0, [0, 100000], 0.1)
        # spec_layout.addWidget(self.widgets['obs_height'], nrow, 2)
        # self.widgets['solar_flag'].stateChanged.connect(
        #     lambda: self.widgets['obs_height'].setDisabled(
        #         not self.widgets['solar_flag'].isChecked()
        #     )
        # )
        # self.widgets['solar_flag'].setChecked(False)

        # nrow += 1

        # Auto-update parameter settings --------------------------------------
        spec_layout.addWidget(QHLine(), nrow, 0, 1, 10)
        nrow += 1

        # Add control for maximum residual
        spec_layout.addWidget(QRightLabel('Good Fit\nResidual Limit'), nrow, 0)
        self.widgets['residual_limit'] = DSpinBox(10, [0, 1000], 0.1)
        spec_layout.addWidget(self.widgets['residual_limit'], nrow, 1)

        # Add control for updating parameters
        self.widgets['update_params'] = QCheckBox('Update Fit\nParameters?')
        spec_layout.addWidget(self.widgets['update_params'], nrow, 2)
        self.widgets['update_params'].setChecked(False)
        self.widgets['update_params'].stateChanged.connect(
            lambda: self.widgets['residual_limit'].setDisabled(
                not self.widgets['update_params'].isChecked()
            )
        )
        nrow += 1

        # Input for fit tolerance ---------------------------------------------
        spec_layout.addWidget(QHLine(), nrow, 0, 1, 10)
        nrow += 1

        spec_layout.addWidget(QRightLabel('Fit tolerance:\n(x1e-8)'), nrow, 0)
        self.widgets['tolerance'] = DSpinBox(1, [0.01, 1e8], 0.01)
        spec_layout.addWidget(self.widgets['tolerance'], nrow, 1)


    def _createLogs(self):
        """Generate program log and control widgets."""
        layout = QGridLayout(self.logFrame)
        layout.setAlignment(Qt.AlignTop)

        # Add button to begin analysis
        self.start_btn = QPushButton('Begin!')
        self.start_btn.clicked.connect(self.control_loop)
        layout.addWidget(self.start_btn, 0, 1)

        # Add button to pause analysis
        self.pause_btn = QPushButton('Pause')
        self.pause_btn.clicked.connect(partial(self.pause_analysis))
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn, 0, 2)

        # Add button to stop analysis
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(partial(self.stop_analysis))
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

        # Add button to add windows
        btn = QPushButton('Add Analysis Window')
        btn.clicked.connect(self.generateNewWindow)
        self.outputTabHolder.setCornerWidget(btn)

        # Setup the fit display ===============================================

        fitTab = QWidget()
        self.outputTabHolder.addTab(fitTab, "Fits")

        layout = QGridLayout(fitTab)
        layout.setAlignment(Qt.AlignTop)

        # Create a control for which window to show
        layout.addWidget(QRightLabel('Plot Window:'), 0, 0)
        self.widgets['plot_window'] = QComboBox()
        layout.addWidget(self.widgets['plot_window'], 0, 1)
        self.widgets['plot_window'].currentTextChanged.connect(
            self.update_fit_plot
        )

        # Set control for plot options
        self.widgets['plot_i0'] = QCheckBox('Show I0?\n(Green)')
        layout.addWidget(self.widgets['plot_i0'], 0, 2)
        self.widgets['plot_bg'] = QCheckBox('Show Background?\n(Purple)')
        layout.addWidget(self.widgets['plot_bg'], 0, 3)
        self.widgets['plot_os'] = QCheckBox('Show Offset?\n(Red)')
        layout.addWidget(self.widgets['plot_os'], 0, 4)

        for cb in [self.widgets[k]for k in ['plot_i0', 'plot_bg', 'plot_os']]:
            cb.stateChanged.connect(lambda: self.update_fit_plot())

        # Generate the graph window
        area = da.DockArea()
        layout.addWidget(area, 1, 0, 1, 6)
        layout.addWidget(QLabel('Wavenumber (cm-1)'), 2, 2)

        # Generate the docks
        d0 = da.Dock('')
        d1 = da.Dock('')
        area.addDock(d0, 'bottom')
        area.addDock(d1, 'bottom')

        # Hide the titlebars
        for d in [d0, d1]:
            d.hideTitleBar()

        # Generate the plot axes
        ax0 = pg.PlotWidget()
        ax1 = pg.PlotWidget()
        ax1.setXLink(ax0)

        # Add plot labels
        ax0.setLabel('left', 'Intensity (counts)')
        ax1.setLabel('left', 'Residual (%)')

        # Add to docks
        d0.addWidget(ax0)
        d1.addWidget(ax1)

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

        # Add legend to first axis
        legend = ax0.addLegend()
        legend.addItem(l0, 'Spectrum')
        legend.addItem(l1, 'Fit')

        self.plot_lines['fit'] = [ax0, ax1]
        self.plot_lines['fit'] = [l0, l1, l2, l3, l4, l5]

        # Setup the initial results graph view ================================

        spectrumTab = QWidget()
        self.outputTabHolder.addTab(spectrumTab, "Spectrum")
        layout = QGridLayout(spectrumTab)
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

        layout.addWidget(QRightLabel('Window'), 0, 1)
        layout.addWidget(QRightLabel('Layer'), 0, 2)
        layout.addWidget(QRightLabel('Gas'), 0, 3)

        # Generate menus to select ratio species
        layout.addWidget(QRightLabel('X-data:'), 1, 0)
        self.widgets['ratio_window_x'] = QComboBox()
        layout.addWidget(self.widgets['ratio_window_x'], 1, 1)
        self.widgets['ratio_window_x'].currentTextChanged.connect(
            lambda: self.update_ratio_window('x')
        )
        self.widgets['ratio_layer_x'] = QComboBox()
        layout.addWidget(self.widgets['ratio_layer_x'], 1, 2)
        self.widgets['ratio_layer_x'].currentTextChanged.connect(
            lambda: self.update_ratio_layer('x')
        )
        self.widgets['ratio_gas_x'] = QComboBox()
        layout.addWidget(self.widgets['ratio_gas_x'], 1, 3)
        self.widgets['ratio_gas_x'].currentTextChanged.connect(
            self.update_ratio_plot
        )

        layout.addWidget(QRightLabel('Y-data:'), 2, 0)
        self.widgets['ratio_window_y'] = QComboBox()
        layout.addWidget(self.widgets['ratio_window_y'], 2, 1)
        self.widgets['ratio_window_y'].currentTextChanged.connect(
            lambda: self.update_ratio_window('y')
        )
        self.widgets['ratio_layer_y'] = QComboBox()
        layout.addWidget(self.widgets['ratio_layer_y'], 2, 2)
        self.widgets['ratio_layer_y'].currentTextChanged.connect(
            lambda: self.update_ratio_layer('y')
        )
        self.widgets['ratio_gas_y'] = QComboBox()
        layout.addWidget(self.widgets['ratio_gas_y'], 2, 3)
        self.widgets['ratio_gas_y'].currentTextChanged.connect(
            self.update_ratio_plot
        )

        # Add displays for the fitted gradient and intercept
        layout.addWidget(QRightLabel('Gradient:'), 1, 4)
        self.ratio_gradient = QLabel('-')
        layout.addWidget(self.ratio_gradient, 1, 5)
        layout.addWidget(QRightLabel('Intercept:'), 2, 4)
        self.ratio_intercept = QLabel('-')
        layout.addWidget(self.ratio_intercept, 2, 5)

        # Add option to include errors in fit
        self.widgets['ratio_fit_errors'] = QCheckBox('Include\nErrors?')
        layout.addWidget(self.widgets['ratio_fit_errors'], 1, 6)
        self.widgets['ratio_fit_errors'].stateChanged.connect(
            self.update_ratio_plot
        )

        # Add option to remove bad fits
        self.widgets['bad_fit_flag'] = QCheckBox('Remove\nBad Fits?')
        layout.addWidget(self.widgets['bad_fit_flag'], 2, 6)
        self.widgets['bad_fit_flag'].stateChanged.connect(
            self.update_ratio_plot
        )

        # Generate the plot
        ratio_area = da.DockArea()
        layout.addWidget(ratio_area, 3, 0, 1, 7)

        # Generate the docks
        ratio_dock = da.Dock('Ratio')
        ratio_area.addDock(ratio_dock, 'top')

        # Generate axes
        ax1 = pg.PlotWidget()
        ratio_dock.addWidget(ax1)

        # Generate plot lines
        l1 = pg.ErrorBarItem()
        l2 = pg.ScatterPlotItem(
            size=10, symbol='x',  pen=pg.mkPen(color=self.PLOTCOLORS[0])
        )
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
        if not dialog.exec_():
            return
        self.addFitWindow(**dialog.info)

    def addFitWindow(self, window):
        """Add a new analysis fit window."""
        # Check if the name exists
        if window in self.windows:
            logger.warning(f'{window} window already exists!')
            return

        # Generate the new tabs
        self.windowTabs[window] = QTabWidget()
        self.outputTabHolder.addTab(self.windowTabs[window], window)

        # Make setup and output tabs
        setupTab = QWidget()
        self.windowTextBoxes[window] = QTextEdit()
        covarTab = QWidget()
        self.windowTabs[window].addTab(setupTab, 'Setup')
        self.windowTabs[window].addTab(self.windowTextBoxes[window], 'Results')
        self.windowTabs[window].addTab(covarTab, 'Covariance')

        self.windowTextBoxes[window].setEnabled(False)

        # Create button to remove the analysis window
        btn = QPushButton('Delete Window')
        btn.clicked.connect(lambda: self.removeFitWindow(window))
        self.windowTabs[window].setCornerWidget(btn)

        # Create widget holder
        winWidgets = Widgets()

        # Inputs tab ==========================================================

        # Generate the layout
        layout = QGridLayout(setupTab)
        layout.setAlignment(Qt.AlignTop)

        # Create inputs for the fit window
        layout.addWidget(QRightLabel('Window Start\n(cm-1):'), 0, 0)
        winWidgets['wn_start'] = SpinBox(0, [0, 1e5])
        layout.addWidget(winWidgets['wn_start'], 0, 1)
        layout.addWidget(QRightLabel('Window End\n(cm-1):'), 0, 2)
        winWidgets['wn_stop'] = SpinBox(1, [0, 1e5])
        layout.addWidget(winWidgets['wn_stop'], 0, 3)

        # Create a checkbox to disable/enable the window
        winWidgets['run_window'] = QCheckBox('Run\nWindow?')
        winWidgets['run_window'].setChecked(True)
        layout.addWidget(winWidgets['run_window'], 0, 4)

        paramTabHolder = QTabWidget()
        layout.addWidget(paramTabHolder, 1, 0, 1, 6)
        # winTab = QScrollArea()
        gasTab = QScrollArea()
        ilsTab = QScrollArea()
        paramTab = QScrollArea()
        # paramTabHolder.addTab(winTab, 'Window Settings')
        paramTabHolder.addTab(gasTab, 'Gas Layers')
        paramTabHolder.addTab(ilsTab, 'ILS Parameters')
        paramTabHolder.addTab(paramTab, 'Other Parameters')

        # Layer parameters ----------------------------------------------------

        playout = QGridLayout(gasTab)
        playout.setAlignment(Qt.AlignTop)

        # Create container objects for layers
        self.layerWidgets[window] = {}

        self.layerTabHolders[window] = QTabWidget()
        self.layerTabHolders[window].setTabsClosable(True)
        self.layerTabHolders[window].tabCloseRequested.connect(
            lambda idx: self.removeLayer(
                window, self.layerTabHolders[window].tabText(idx)
            )
        )
        playout.addWidget(self.layerTabHolders[window], 1, 0)

        # Create a button to add a layer
        btn = QPushButton('Add layer')
        btn.clicked.connect(lambda: self.generateNewLayer(window))
        self.layerTabHolders[window].setCornerWidget(btn)
        self.layerTabHolders[window].cornerWidget().setMinimumSize(20, 25)

        # ILS parameters ------------------------------------------------------

        playout = QGridLayout(ilsTab)
        playout.setAlignment(Qt.AlignTop)

        # Input for apodization function
        playout.addWidget(QRightLabel('Apodization\nFunction'), 0, 0)
        winWidgets['apod_function'] = QComboBox()
        winWidgets['apod_function'].addItems(
            ['NB_weak', 'NB_medium', 'NB_strong', 'triangular', 'boxcar']
        )
        winWidgets.set('apod_function', 'NB_medium')
        playout.addWidget(winWidgets['apod_function'], 0, 1, 1, 2)

        playout.addWidget(QHLine(), 1, 0, 1, 5)

        playout.addWidget(QRightLabel('Value'), 2, 1)
        playout.addWidget(QRightLabel('Low Bound'), 2, 2)
        playout.addWidget(QRightLabel('High Bound'), 2, 3)

        # Input for fov
        playout.addWidget(QRightLabel('Field of View\n(m.rad)'), 3, 0)
        winWidgets['fov'] = DSpinBox(10, [0, 100], 0.1)
        playout.addWidget(winWidgets['fov'], 3, 1)
        winWidgets['fov_lo_bound'] = DSpinBox(10, [0, 100], 0.1)
        playout.addWidget(winWidgets['fov_lo_bound'], 3, 2)
        winWidgets['fov_hi_bound'] = DSpinBox(40, [0, 100], 0.1)
        playout.addWidget(winWidgets['fov_hi_bound'], 3, 3)
        winWidgets['fit_fov'] = QCheckBox('Fit?')
        winWidgets['fit_fov'].setChecked(False)
        playout.addWidget(winWidgets['fit_fov'], 3, 4)

        # Input for OPD
        playout.addWidget(QRightLabel('Optical Path\nDifference (cm)'), 4, 0)
        winWidgets['opd'] = DSpinBox(1.6, [0, 100], 0.01)
        playout.addWidget(winWidgets['opd'], 4, 1)
        winWidgets['opd_lo_bound'] = DSpinBox(10, [0, 100], 0.1)
        playout.addWidget(winWidgets['opd_lo_bound'], 4, 2)
        winWidgets['opd_hi_bound'] = DSpinBox(40, [0, 100], 0.1)
        playout.addWidget(winWidgets['opd_hi_bound'], 4, 3)
        winWidgets['fit_opd'] = QCheckBox('Fit?')
        winWidgets['fit_opd'].setChecked(False)
        playout.addWidget(winWidgets['fit_opd'], 4, 4)

        # Other parameters ----------------------------------------------------

        playout = QGridLayout(paramTab)
        playout.setAlignment(Qt.AlignTop)

        # Background n params and apriori
        playout.addWidget(QRightLabel('Num. Background\nParams'), 0, 0)
        winWidgets['n_bg_poly'] = SpinBox(1, [1, 100])
        playout.addWidget(winWidgets['n_bg_poly'], 0, 1)
        playout.addWidget(QRightLabel('Apriori\nBackground'), 0, 2)
        winWidgets['bg_poly_apriori'] = DSpinBox(0, [0, 1e30])
        playout.addWidget(winWidgets['bg_poly_apriori'], 0, 3)

        # Background n params

        # Shift n params and apriori
        playout.addWidget(QRightLabel('Num. Shift\nParams'), 1, 0)
        winWidgets['n_shift'] = SpinBox(1, [0, 100])
        playout.addWidget(winWidgets['n_shift'], 1, 1)
        playout.addWidget(QRightLabel('Apriori\nShift'), 1, 2)
        winWidgets['shift_apriori'] = DSpinBox(0, [-1000, 1000])
        playout.addWidget(winWidgets['shift_apriori'], 1, 3)
        winWidgets['fit_shift'] = QCheckBox('Fit?')
        playout.addWidget(winWidgets['fit_shift'], 1, 4)
        winWidgets['fit_shift'].setChecked(True)

        # Offset n params and apriori
        playout.addWidget(QRightLabel('Num. Offset\nParams'), 2, 0)
        winWidgets['n_offset'] = SpinBox(0, [0, 100])
        playout.addWidget(winWidgets['n_offset'], 2, 1)
        playout.addWidget(QRightLabel('Apriori\nOffset'), 2, 2)
        winWidgets['offset_apriori'] = DSpinBox(0, [-1000, 1000])
        playout.addWidget(winWidgets['offset_apriori'], 2, 3)
        winWidgets['fit_offset'] = QCheckBox('Fit?')
        playout.addWidget(winWidgets['fit_offset'], 2, 4)
        winWidgets['fit_offset'].setChecked(False)

        # Output graphs =======================================================

        # Add fit regions to main spectrum plot and connect to the wavenumber
        # bounds
        self.plot_regions[window] = pg.LinearRegionItem([0, 0])
        self.plot_regions[window].setMovable(False)
        # self.plot_regions[window].setToolTip
        winWidgets['wn_start'].valueChanged.connect(
            lambda: self.plot_regions[window].setRegion(
                [winWidgets.get('wn_start'), winWidgets.get('wn_stop')]
            )
        )
        winWidgets['wn_stop'].valueChanged.connect(
            lambda: self.plot_regions[window].setRegion(
                [winWidgets.get('wn_start'), winWidgets.get('wn_stop')]
            )
        )
        self.plot_axes['main'][0].addItem(self.plot_regions[window])

        # Gas covariance plot =================================================

        # Generate plot area
        clayout = QGridLayout(covarTab)
        cov_area = da.DockArea()
        clayout.addWidget(cov_area, 0, 0)

        # Generate the docks
        cov_dock = da.Dock('Covariance')
        cov_area.addDock(cov_dock, 'top')

        # Generate axes
        plot = pg.PlotItem()
        im_view = pg.ImageView(view=plot)
        im_view.setColorMap(pg.colormap.get('plasma'))
        cov_dock.addWidget(im_view)
        self.cov_plot[window] = plot
        self.cov_imview[window] = im_view

        # Add to overall widgets ==============================================

        self.windowWidgets[window] = winWidgets

        self.windows.append(window)

        # Update plot options
        state = self.widgets.get('plot_window')
        self.widgets['plot_window'].clear()
        self.widgets['plot_window'].addItems(self.windows)
        if state in self.windows:
            self.widgets.set('plot_window', state)

        # Update ratio boxes ==================================================

        # Get the current state
        xwin = self.widgets.get('ratio_window_x')
        ywin = self.widgets.get('ratio_window_y')

        # Clear the box
        self.widgets['ratio_window_x'].clear()
        self.widgets['ratio_window_y'].clear()

        # Update the list
        self.widgets['ratio_window_x'].addItems(self.windows)
        self.widgets['ratio_window_y'].addItems(self.windows)

        # Reset to origional state
        self.widgets.set('ratio_window_x', xwin)
        self.widgets.set('ratio_window_y', ywin)

        logger.info(f'{window} window added')

    def removeFitWindow(self, window):
        """Remove fit window."""
        if window not in self.windowTabs:
            return

        # Get the index of the window tab
        window_idx = list(self.windowTabs.keys()).index(window)

        # Remove the window tab from the GUI
        self.outputTabHolder.removeTab(window_idx + 3)

        # Delete the actual widget from memory
        self.windowTabs[window].setParent(None)

        # Remove window from main plot
        self.plot_axes['main'][0].removeItem(self.plot_regions[window])

        # Remove from list of windows
        self.windows.remove(window)
        self.windowTabs.pop(window)
        self.windowWidgets.pop(window)
        self.cov_plot.pop(window)
        self.cov_imview.pop(window)
        self.plot_regions.pop(window)

        # Update ratio boxes
        self.refresh_ratio_windows()

        logger.info(f'{window} window removed')

    # =========================================================================
    # Add new gas layer
    # =========================================================================

    def generateNewLayer(self, window):
        """Get a new window name from a popup and add it."""
        # Run new layer wizard
        dialog = NewLayerWizard(self)
        if not dialog.exec_():
            return
        self.addLayer(window, **dialog.info)

    def addLayer(self, window, layer_id):
        """."""
        # Check if the name exists
        if layer_id in self.layerWidgets[window]:
            logger.warning(f'{layer_id} layer already exists!')
            return

        # Generate the new layer tab
        layerTab = QWidget()
        self.layerTabHolders[window].addTab(layerTab, layer_id)
        layout = QGridLayout(layerTab)
        layout.setAlignment(Qt.AlignTop)

        # Create widget holder
        layerWidgets = Widgets()
        self.layerWidgets[window][layer_id] = layerWidgets

        # Layer settings ======================================================

        nrow = 0

        layout.addWidget(QRightLabel('Value'), nrow, 1)
        layout.addWidget(QRightLabel('Low Bound'), nrow, 2)
        layout.addWidget(QRightLabel('High Bound'), nrow, 3)
        nrow += 1

        # Input for temperature
        layout.addWidget(QRightLabel('Temperature (K):'), nrow, 0)
        layerWidgets['temperature'] = DSpinBox(298, [0, 2000], 0.1)
        layout.addWidget(layerWidgets['temperature'], nrow, 1)
        layerWidgets['temperature_lo_bound'] = DSpinBox(273, [0, 2000], 0.1)
        layout.addWidget(layerWidgets['temperature_lo_bound'], nrow, 2)
        layerWidgets['temperature_hi_bound'] = DSpinBox(373, [0, 2000], 0.1)
        layout.addWidget(layerWidgets['temperature_hi_bound'], nrow, 3)
        layerWidgets['vary_temperature'] = QCheckBox('Fit?')
        layerWidgets['vary_temperature'].setChecked(False)
        layout.addWidget(layerWidgets['vary_temperature'], nrow, 4)
        nrow += 1

        layout.addWidget(QHLine(), nrow, 0, 1, 5)
        nrow += 1

        # Input for pressure
        layout.addWidget(QRightLabel('Pressure (mb):'), nrow, 0)
        layerWidgets['pressure'] = DSpinBox(1013, [0, 10000], 0.1)
        layout.addWidget(layerWidgets['pressure'], nrow, 1)

        # Input for path length
        layout.addWidget(QRightLabel('Path Length (m):'), nrow, 2)
        layerWidgets['path_length'] = SpinBox(100, [0, 10000000])
        layout.addWidget(layerWidgets['path_length'], nrow, 3)
        nrow += 1

        layout.addWidget(QHLine(), nrow, 0, 1, 5)
        nrow += 1

        # Add gas input table
        layerWidgets['gases'] = paramTable(
            layerTab, 'param', width=200, gas_list=self.gas_list.keys()
        )
        layout.addWidget(layerWidgets['gases'], nrow, 0, 1, 5)

    def removeLayer(self, window, layer_id):
        """Remove layer from analysis window."""
        if layer_id not in self.layerWidgets[window]:
            return

        # Get the index of the layer tab
        layer_idx = list(self.layerWidgets[window].keys()).index(layer_id)

        # Delete the tab
        self.layerTabHolders[window].removeTab(layer_idx)

        # Remove from GUI dictionaries
        self.layerWidgets[window].pop(layer_id)

        # Update ratio boxes
        self.update_ratio_layer('x')
        self.update_ratio_layer('y')

    # =========================================================================
    # Update ratio plot inputs
    # =========================================================================

    def refresh_ratio_windows(self):
        x_window = self.widgets.get('ratio_window_x')
        y_window = self.widgets.get('ratio_window_y')

        self.widgets['ratio_window_x'].clear()
        self.widgets['ratio_window_x'].addItems(self.windows)
        self.widgets['ratio_window_y'].clear()
        self.widgets['ratio_window_y'].addItems(self.windows)

        if x_window in self.windows:
            self.widgets.set('ratio_window_x', x_window)
        if y_window in self.windows:
            self.widgets.set('ratio_window_y', x_window)

    def update_ratio_window(self, axis):
        """Update ratio selection comboboxes."""
        try:

            # Get current window and layer values for the gien axis
            window = self.widgets.get(f'ratio_window_{axis}')
            layer = self.widgets.get(f'ratio_layer_{axis}')

            # Make sure a window is selected
            if window != '':

                # Clear and reset the layer list (if the layer is still there)
                self.widgets[f'ratio_layer_{axis}'].clear()
                layer_list = list(self.layerWidgets[window].keys())
                self.widgets[f'ratio_layer_{axis}'].addItems(layer_list)
                if layer in layer_list:
                    self.widgets.set(f'ratio_layer_{axis}', layer)

        except KeyError:
            pass

    def update_ratio_layer(self, axis):
        """Update ratio selection comboboxes."""
        try:

            # Get current window, layer and gas values for the gien axis
            window = self.widgets.get(f'ratio_window_{axis}')
            layer = self.widgets.get(f'ratio_layer_{axis}')
            gas = self.widgets.get(f'ratio_gas_{axis}')

            # Make sure a window and layer is selected
            if window != '' and layer != '':

                # Clear and reset the gas list
                self.widgets[f'ratio_gas_{axis}'].clear()
                gas_list = list(
                    self.layerWidgets[window][layer]['gases'].getData().keys()
                )
                self.widgets[f'ratio_gas_{axis}'].addItems(gas_list)
                if gas in gas_list:
                    self.widgets.set(f'ratio_gas_{axis}', gas)

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
        mainWidgets = widgetData['main_settings']

        self.update_status('Initialising')

        # Ensure the output directory exists
        if not os.path.isdir(mainWidgets['save_dir']):
            os.makedirs(mainWidgets['save_dir'])

        # Generate a log file handler for this analysis loop
        self.analysis_logger = logging.FileHandler(
            f'{mainWidgets["save_dir"]}/ftpyr_analysis.log',
            mode='w'
        )
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_fmt = '%Y-%m-%d %H:%M:%S'
        f_format = logging.Formatter(log_fmt, date_fmt)
        self.analysis_logger.setFormatter(f_format)
        self.analysis_logger.setLevel(logging.DEBUG)
        logger.addHandler(self.analysis_logger)

        # Form the output file name
        self.outfname = f"{mainWidgets['save_dir']}/all_gas_output.csv"

        # Generate the setup worker
        setupWorker = SetupWorker(widgetData, self.outfname)
        setupWorker.signals.error.connect(self.update_error)
        setupWorker.signals.initialize.connect(self.initialize_window)
        setupWorker.signals.finished.connect(self.begin_analysis)
        self.threadpool.start(setupWorker)

    def initialize_window(self, window, analyser):
        """Initialize window analyser and results table."""
        # Add the analyser to the dictionary
        self.analysers[window] = analyser

    def begin_analysis(self):
        """Run main analysis loop."""
        # Check initialisation went ok
        if self.initialisation_error_flag:
            logger.info('Error with window initialisation')
            return
        logger.info('All windows initialised, begining analysis loop')

        # Pull the widget data
        mainWidgetData = self.getWidgetData()['main_settings']

        # Get whether running in real time or post-analysis
        # self.analysis_mode = self.widgets['analysis_mode'].currentText()
        self.analysis_mode = 'Post-Process'

        # Get the spectra to analyse
        if self.analysis_mode == 'Post-Process':
            self.spectra_list = mainWidgetData['spec_fnames'].split('\n')
        else:
            watch_dir = self.widgets.get('watch_dir')
            files = os.listdir(watch_dir)
            self.spectra_list = [f'{watch_dir}/{file}' for file in files]

        # Create a spectrum counter
        self.spectrum_counter = 0

        # Create dictionary to hold fit results
        self.fit_results = {}

        # Get the output ppmm flag and disable the option
        self.output_ppmm_flag = mainWidgetData['output_units'] == 'ppm.m'

        self.update_status('Analysing')

        # Generate the thread workers
        for window in self.windows:

            # Generate the worker
            analysisWorker = AnalysisWorker(window, self.analysers[window])
            analysisWorker.signals.results.connect(self.get_results)
            analysisWorker.signals.error.connect(self.update_error)
            self.analysisWorkers[window] = analysisWorker
            self.ready_flags[window] = False
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
        for worker in self.analysisWorkers.values():
            worker.stop()
        logger.debug('Analysis stopped')
        self.analysis_complete()

    def pause_analysis(self):
        """Pause the workers."""
        # Pause each worker
        for worker in self.analysisWorkers.values():
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
        window, fit = results

        # Add to the dictionary
        self.fit_results[window] = fit

        if fit is not None:
            self.update_fit_plot()

        # Update results text output
        self.windowTextBoxes[window].setText(fit.params.__repr__())

        # Update worker ready flag
        self.ready_flags[window] = True

        # Check if all workers are ready
        ready_flags = np.array([b for b in self.ready_flags.values()])
        if ready_flags.all():

            # Write the results
            with open(self.outfname, 'a') as outfile:

                # Write the filename and timestamp
                ts = self.spectrum.attrs['timestamp']
                outfile.write(f'{self.spec_filename},{ts}')

                # Write the gas parameter results
                for window in self.ready_flags.keys():
                    fit = self.fit_results[window]

                    for par in fit.params.get_all_parameters().values():
                        if par.layer_id is not None:
                            layer = fit.params.layers[par.layer_id]
                            val = layer.get_fit_value(par.name)
                            err = layer.get_fit_error(par.name)
                            outfile.write(f',{val},{err}')

                    # Write the fit quality result
                    outfile.write(
                        f',{fit.nerr},'
                        f'{fit.data.residual.data.max()},'
                        f'{fit.data.residual.data.std()}'
                    )
                outfile.write('\n')

            # For Post-Processing
            if self.analysis_mode == 'Post-Process':
                # If all spectra have been analysed, stop the analysis workers
                if self.spectrum_counter == len(self.spectra_list):
                    self.stop_analysis()

                # If not, analyse the next spectrum
                else:
                    self.set_next_spectrum()

            # For Real-Time processing
            else:
                self.fileTimer = QTimer(self)
                self.fileTimer.setInterval(100)
                self.fileTimer.timeout.connect(self.check_for_next_spectrum)
                self.fileTimer.start()

    def check_for_next_spectrum(self):
        """."""

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
        x = self.spectrum.coords['wavenumber'].to_numpy()
        y = self.spectrum.to_numpy()
        self.plot_lines['main'][0].setData(x, y)

    def update_fit_plot(self):
        """Update the window results graphs and table."""
        try:
            # Pull the window results
            window = self.widgets.get('plot_window')
            fit = self.fit_results[window]
        except KeyError:
            return

        # Update the fit plot
        self.plot_lines['fit'][0].setData(
            fit.data.wavenumber, fit.data.spectrum
        )
        self.plot_lines['fit'][1].setData(fit.data.wavenumber, fit.data.fit)

        # Add optional lines
        if self.widgets.get('plot_i0'):
            self.plot_lines['fit'][2].setData(
                fit.data.wavenumber,
                fit.data.bg_poly + np.nan_to_num(fit.data.offset)
            )
        else:
            self.plot_lines['fit'][2].setData([], [])
        if self.widgets.get('plot_bg'):
            self.plot_lines['fit'][3].setData(
                fit.data.wavenumber, fit.data.bg_poly
            )
        else:
            self.plot_lines['fit'][3].setData([], [])
        if self.widgets.get('plot_os'):
            self.plot_lines['fit'][4].setData(
                fit.data.wavenumber, fit.data.offset
            )
        else:
            self.plot_lines['fit'][4].setData([], [])

        # Add residual
        self.plot_lines['fit'][5].setData(
            fit.data.wavenumber, fit.data.residual
        )

        # Update covariance plot
        # gases = [gas for gas in fit.params.extract_gases().keys()]
        # ngases = len(gases)
        # tick_pos = np.arange(ngases) + 0.5
        # cov_data = np.log10(np.sqrt(abs(fit.pcov[:ngases, :ngases])))
        # try:
        #     self.cov_imview[window].setImage(cov_data)
        # except Exception:
        #     self.cov_imview[window].setImage(np.zeros([ngases, ngases]))
        # yax = self.cov_plot[window].getAxis('left')
        # yax.setTicks([[(tick_pos[i], gases[i]) for i in range(ngases)]])
        # xax = self.cov_plot[window].getAxis('bottom')
        # xax.setTicks([[(tick_pos[i], gases[i]) for i in range(ngases)]])

    def update_ratio_plot(self):
        """Update the data shown on the ratio plot."""
        # Read in the time series results for the ratio plots
        try:
            df = pd.read_csv(self.outfname, parse_dates=['Timestamp'])
        except AttributeError:
            return
        xwin = self.widgets.get('ratio_window_x')
        xlayer = self.widgets.get('ratio_layer_x')
        xgas = self.widgets.get('ratio_gas_x')
        ywin = self.widgets.get('ratio_window_y')
        ylayer = self.widgets.get('ratio_layer_y')
        ygas = self.widgets.get('ratio_gas_y')

        try:
            # Remove bad fits if desired
            if self.widgets.get('bad_fit_flag'):
                idx = np.logical_and(
                    df[f'FitQuality ({xwin})'] == 0,
                    df[f'FitQuality ({ywin})'] == 0
                )
                df = df[idx]

            # Unpack good fit values and errors
            xval = df[f'{xgas} ({xwin}/{xlayer})'].to_numpy()
            xerr = df[f'{xgas}_err ({xwin}/{xlayer})'].to_numpy()
            yval = df[f'{ygas} ({ywin}/{ylayer})'].to_numpy()
            yerr = df[f'{ygas}_err ({ywin}/{ylayer})'].to_numpy()

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
                    myodr = odr.ODR(
                        data=odr.RealData(x=xval, y=yval),
                        model=odr.unilinear,
                        beta0=[m, c]
                    ).run()
                    popt = myodr.beta
                    perr = myodr.sd_beta

                # Fit with taking errors into account
                else:
                    myodr = odr.ODR(
                        data=odr.RealData(x=xval, y=yval, sx=xerr, sy=yerr),
                        model=odr.unilinear,
                        beta0=[m, c]
                    ).run()
                    popt = myodr.beta
                    perr = myodr.sd_beta

                # Make the best fit line
                yfit = lin_fit(xfit, *popt)

                # Update the plots
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
        # Get the analysis mode
        # analysis_mode = self.widgets['analysis_mode'].currentText()
        analysis_mode = 'Post-Process'

        try:
            # Get the first listed file, either from the input or the watched
            # directory
            if analysis_mode == 'Post-Process':
                filename = self.widgets.get('spec_fnames').split('\n')[0]
            else:
                fpath = self.widgets.get('watch_dir')
                files = os.listdir(fpath)
                filename = f'{fpath}/{files[0]}'

            # Plot if it has changed
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

    def flip_mode(self):
        """Flip inputs on analysis mode."""
        # analysis_mode = self.widgets['analysis_mode'].currentText()
        analysis_mode = 'Post-Process'
        if analysis_mode == 'Post-Process':
            self.widgets['watch_dir'].setEnabled(False)
            self.widgets['spec_fnames'].setReadOnly(False)
            self.widgets['spec_fnames'].setEnabled(True)
            self.plot_first_spectrum()
        else:
            self.widgets['watch_dir'].setEnabled(True)
            self.widgets['spec_fnames'].setReadOnly(True)
            self.widgets['spec_fnames'].setEnabled(False)
            self.plot_first_spectrum()

    # =========================================================================
    # Program settings and theme
    # =========================================================================

    def getWidgetData(self):
        """Get the widget data into a single dictionary."""
        # Pull the main widget data
        widgetData = {
            'theme': self.theme,
            'main_settings': self.widgets.get_values(),
            'window_settings': {
                **{
                    key: {
                        **widgets.get_values(),
                        'layers': {
                            layer: self.layerWidgets[key][layer].get_values()
                            for layer in self.layerWidgets[key]
                        }
                    }
                    for key, widgets in self.windowWidgets.items()
                }
            }
        }

        return widgetData

    def saveConfig(self, asksavepath=True):
        """Save the program configuration."""
        # Pull the main widget data
        config = self.getWidgetData()

        # Get save filename if required
        if asksavepath or self.config_fname is None:
            filter = 'YAML (*.yml *.yaml);;All Files (*)'
            fname, s = QFileDialog.getSaveFileName(
                self, 'Save Config', '', filter
            )
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
            fname, _ = QFileDialog.getOpenFileName(
                self, 'Load Config', '', filter
            )

        # Open the config file
        try:
            with open(fname, 'r') as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.FullLoader)

            logger.info(f'Loading config from {self.config_fname}')

            # Clear current windows
            for window in self.windowWidgets:
                self.removeFitWindow(window)

                # Clear all layers
                for layer_id in self.layerWidgets[window]:
                    self.removeLayer(window, layer_id)

            # Add windows and layers
            for window, window_data in config['window_settings'].items():
                data = {
                    key: val for key, val in window_data.items()
                    if key != 'layers'
                }
                layer_data = window_data['layers']
                self.addFitWindow(window)
                self.windowWidgets[window].set_values(data)

                for layer_id, data in layer_data.items():
                    self.addLayer(window, layer_id)
                    self.layerWidgets[window][layer_id].set_values(data)

            # Apply main gui settings
            self.widgets.set_values(config['main_settings'])
            if 'theme' in config:
                self.theme = config['theme']

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

    def __init__(self, widgetData, outfname, *args, **kwargs):
        """Initialise."""
        super(SetupWorker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
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
            main_settings = self.widgetData['main_settings']
            window_settings = self.widgetData['window_settings']
            for window, settings in window_settings.items():

                # Setup window parameters
                params = Parameters()

                # Setup gas layers
                for layer_id, layer_data in settings['layers'].items():

                    # Create the layer
                    layer = Layer(
                        layer_id=layer_id,
                        temperature=layer_data['temperature'],
                        pressure=layer_data['pressure'],
                        path_length=layer_data['path_length'],
                        vary_temperature=layer_data['vary_temperature'],
                        temperature_bounds=[
                            layer_data['temperature_lo_bound'],
                            layer_data['temperature_hi_bound']
                        ],
                        temperature_step=1
                    )

                    # Add gases
                    for gas, args in layer_data['gases'].items():
                        layer.add_gas(gas, **args)

                        main_outfile.write(
                            f',{gas} ({window}/{layer_id})'
                            f',{gas}_err ({window}/{layer_id})'
                        )

                    # Add layer to parameters
                    params.add_layer(layer)

                main_outfile.write(
                    f',FitQuality ({window}),MaxResidual ({window}),'
                    f'StdevResidual ({window})'
                )

                # Add other parameters in the fit

                # Add background parameters
                for i in range(settings['n_bg_poly']):
                    if i == 0:
                        value = settings['bg_poly_apriori']
                    else:
                        value = 0
                    params.add(name=f'bg_poly{i}', value=value)

                # Add shift parameters
                for i in range(settings['n_shift']):
                    if i == 0:
                        value = settings['shift_apriori']
                    else:
                        value = 0
                    params.add(
                        name=f'shift{i}', value=value,
                        vary=settings['fit_shift']
                    )

                # Add offset parameters
                for i in range(settings['n_offset']):
                    if i == 0:
                        value = settings['offset_apriori']
                    else:
                        value = 0
                    params.add(name=f'offset{i}', value=value,
                               vary=settings['fit_offset'])

                # Add ILS parameters
                params.add(
                    name='fov',
                    value=settings['fov'],
                    vary=settings['fit_fov']
                )
                params.add(
                    name='opd',
                    value=settings['opd'],
                    vary=settings['fit_opd']
                )

                # Setup analyser settings
                logger.info(f'Generating analyser for {window} window')
                outfile = f"{main_settings['save_dir']}/{window}_output.csv"
                output_ppmm_flag = main_settings['output_units'] == 'ppm.m'
                analyser_settings = {
                    'params': params,
                    'rfm_path': main_settings['rfm_path'],
                    'hitran_path': main_settings['hitran_path'],
                    'wn_start': settings['wn_start'],
                    'wn_stop': settings['wn_stop'],
                    'zero_fill_factor': main_settings['zero_fill_factor'],
                    # 'solar_flag': self.widgetData['solar_flag'],
                    # 'obs_height': self.widgetData['obs_height'],
                    'update_params': main_settings['update_params'],
                    'residual_limit': main_settings['residual_limit'],
                    'pts_per_cm': main_settings['pts_per_cm'],
                    'model_padding': main_settings['model_padding'],
                    'apod_function': settings['apod_function'],
                    'tolerance': main_settings['tolerance'] * 1e-8,
                    'outfile': outfile,
                    'output_ppmm_flag': output_ppmm_flag,
                    'progress_bars': False
                }

                # Log analyser settings
                logger.debug('Analyser settings:')
                for key, value in analyser_settings.items():
                    logger.debug(f'{key}: {value}')

                # Generate the analyser function
                self.analyser = Analyser(**analyser_settings)

                # Setup the window in the front end
                self.signals.initialize.emit(window, self.analyser)

            # Start a new line from the header
            main_outfile.write('\n')


class AnalysisWorker(QRunnable):
    """Worker class to handle spectra analysis in a separate thread."""

    def __init__(self, window, analyser, *args, **kwargs):
        """Initialise."""
        super(AnalysisWorker, self).__init__()
        self.window = window
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
        logger.info(f'{self.window} analyser finished')

    def fn(self):
        """Main analysis loop."""
        while not self.isStopped:
            if self.spectrum is not None and not self.isPaused:
                fit = self.analyser.fit(self.spectrum)
                self.spectrum = None
                self.signals.results.emit([self.window, fit])
            else:
                time.sleep(0.001)

    def pause(self):
        """Pause the analysis."""
        if self.isPaused:
            logger.debug(f'{self.window} analyser played')
            self.isPaused = False
        else:
            logger.debug(f'{self.window} analyser paused')
            self.isPaused = True

    def stop(self):
        """Stop the analysis."""
        logger.debug(f'{self.window} analyser stopped')
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
        self.setWindowTitle('New fit window')

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
            'window': str(self.wname.text())}
        self.accept()

    def cancel_action(self):
        """Close the window without creating a new station."""
        self.info = {}
        self.close()

# =============================================================================
# New fit window wizard
# =============================================================================

class NewLayerWizard(QDialog):
    """Opens a wizard to define a new station."""

    def __init__(self, parent=None):
        """Initialise the window."""
        super(NewLayerWizard, self).__init__(parent)

        # Set the window properties
        self.setWindowTitle('New layer')

        self._createApp()

    def _createApp(self):
        # Set the layout
        layout = QGridLayout()
        self.layer_id = QLineEdit()
        layout.addWidget(QLabel('Name:'), 0, 0)
        layout.addWidget(self.layer_id, 0, 1)

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
            'layer_id': str(self.layer_id.text())}
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
        if isinstance(self[key], QTextEdit):
            return self[key].toPlainText()
        elif isinstance(self[key], QLineEdit):
            return self[key].text()
        elif isinstance(self[key], QComboBox):
            return str(self[key].currentText())
        elif isinstance(self[key], QCheckBox):
            return self[key].isChecked()
        elif isinstance(self[key], (QSpinBox, QDoubleSpinBox)):
            return self[key].value()
        elif isinstance(self[key], paramTable):
            return self[key].getData()
        else:
            raise ValueError('Widget type not recognised!')

    def set(self, key, value):
        """Set the value of a widget."""
        if key not in self.keys():
            logger.warning(f'{key} widget not found!')
        elif isinstance(self[key], (QTextEdit, QLineEdit)):
            self[key].setText(str(value))
        elif isinstance(self[key], QComboBox):
            index = self[key].findText(value, Qt.MatchFixedString)
            if index >= 0:
                self[key].setCurrentIndex(index)
        elif isinstance(self[key], QCheckBox):
            self[key].setChecked(value)
        elif isinstance(self[key], (QSpinBox, SpinBox)):
            self[key].setValue(int(value))
        elif isinstance(self[key], (QDoubleSpinBox, DSpinBox)):
            self[key].setValue(float(value))
        elif isinstance(self[key], paramTable):
            return self[key].setData(value)
        else:
            raise ValueError('Widget type not recognised!')

    def get_values(self):
        return {key: self.get(key) for key in self.keys()}

    def set_values(self, widget_values):
        for key, val in widget_values.items():
            self.set(key, val)


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
        self.setColumnCount(2)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Species', 'Vary?'])
        header = self.horizontalHeader()
        for i in range(2):
            header.setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents
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
            dm = QComboBox()
            dm.addItems(self.gas_list)
            self.setCellWidget(n, 0, dm)
            cb = QCheckBox()
            cb.setChecked(True)
            self.setCellWidget(n, 1, cb)

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
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == addAction:
            self.add_row()
        if action == remAction:
            self.rem_row()

    def setData(self, data):
        """Populate the table using saved config."""
        for i, (key, values) in enumerate(data.items()):
            self.add_row()

            index = self.cellWidget(i, 0).findText(key, Qt.MatchFixedString)
            if index >= 0:
                self.cellWidget(i, 0).setCurrentIndex(index)
            self.cellWidget(i, 1).setChecked(values['vary'])

    def getData(self):
        """Extract the information from the table."""
        try:
            # Read the data from a param table
            data = {
                self.cellWidget(i, 0).currentText(): {
                    'vary': self.cellWidget(i, 1).isChecked()
                }
                for i in range(self.rowCount())
            }
        except AttributeError:
            data = {}

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
    sys.exit(app.exec_())
