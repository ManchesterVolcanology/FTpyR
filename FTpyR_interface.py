"""Main script for the FTpyR user interface."""
import os
import sys
import yaml
import time
import logging
import traceback
import qdarktheme
import numpy as np
import pyqtgraph as pg
from functools import partial
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import (Qt, QObject, QThread, Signal, Slot,
                            QThreadPool, QRunnable)
from PySide6.QtWidgets import (QMainWindow, QScrollArea, QGridLayout,
                               QApplication, QToolBar, QFrame, QSplitter,
                               QProgressBar, QLabel, QLineEdit, QTextEdit,
                               QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
                               QPlainTextEdit, QPushButton, QFileDialog,
                               QWidget, QTabWidget, QDialog, QHBoxLayout,
                               QTableWidget, QTableWidgetItem, QMenu)

from ftpyr.read import read_spectrum


__version__ = '0.1.0'
__author__ = 'Ben Esse'

# =============================================================================
# =============================================================================
# Setup logging
# =============================================================================
# =============================================================================

# Connect to the logger
logger = logging.getLogger(__name__)


class Signaller(QObject):
    """Signaller object for logging from QThreads."""
    signal = Signal(str, logging.LogRecord)


class QtHandler(logging.Handler):
    """Handler object for handling logs from QThreads."""

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
        """.Initialise the main window."""
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
        self.graphwins = {}
        self.windows = []
        self.plot_axes = {}
        self.plot_lines = {}
        self.plot_regions = {}

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

        filetab = QWidget()
        self.inputTabHolder.addTab(filetab, 'Global Setup')
        file_layout = QGridLayout(filetab)

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
        btn = QPushButton('Browse')
        btn.setFixedSize(70, 25)
        btn.clicked.connect(
            partial(self.browse, self.widgets['spec_fnames'], 'multi', None)
        )
        file_layout.addWidget(btn, 1, 4)

        # Add an input for the RFM exe path
        file_layout.addWidget(QLabel('RFM:'), 2, 0)
        self.widgets['rfm_path'] = QLineEdit()
        self.widgets['rfm_path'].setToolTip('RFM executable file')
        file_layout.addWidget(self.widgets['rfm_path'], 2, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.setFixedSize(70, 25)
        btn.clicked.connect(
            partial(self.browse, self.widgets['rfm_path'], 'single', None)
        )
        file_layout.addWidget(btn, 2, 4)

        # Add an input for the HITRAN database path
        file_layout.addWidget(QLabel('HITRAN:'), 3, 0)
        self.widgets['hitran_path'] = QLineEdit()
        self.widgets['hitran_path'].setToolTip('HITRAN database file')
        file_layout.addWidget(self.widgets['hitran_path'], 3, 1, 1, 3)
        btn = QPushButton('Browse')
        btn.setFixedSize(70, 25)
        btn.clicked.connect(
            partial(self.browse, self.widgets['hitran_path'], 'single', None)
        )
        file_layout.addWidget(btn, 3, 4)

    def _createLogs(self):
        """Generate program log widgets."""
        layout = QGridLayout(self.logFrame)

        # Add button to begin analysis
        self.start_btn = QPushButton('Begin!')
        self.start_btn.setToolTip('Begin spectra analysis')
        self.start_btn.clicked.connect(self.begin_analysis)
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
        layout.addWidget(self.logBox, 2, 0, 1, 5)
        logger.info(f'Welcome to FTpyR v{__version__}! Written by Ben Esse')

    def _createOutputs(self):
        """Generate output and display widgets."""
        layout = QGridLayout(self.outputFrame)
        layout.setAlignment(Qt.AlignTop)

        # Form the tab widget
        self.outputTabHolder = QTabWidget()
        layout.addWidget(self.outputTabHolder, 0, 0)

        # Setup the initial results graph view ================================

        graphtab = QWidget()
        self.outputTabHolder.addTab(graphtab, "Overview")
        glayout = QGridLayout(graphtab)
        self.graphwins['main'] = pg.GraphicsLayoutWidget(show=True)
        pg.setConfigOptions(antialias=True)
        glayout.addWidget(self.graphwins['main'], 0, 0)

        # Make the graphs =====================================================

        # Generate axes
        ax0 = self.graphwins['main'].addPlot(  # Full spectrum
            row=0, col=0, colspan=2
        )
        ax1 = self.graphwins['main'].addPlot(row=1, col=0)  # Ratio plot 1
        ax2 = self.graphwins['main'].addPlot(row=1, col=1)  # Ratio plot 2

        # Generate plot lines
        pen0 = pg.mkPen(color=self.PLOTCOLORS[0], width=1.0)
        l0 = ax0.plot(pen=pen0)

        # Store graph objects
        self.plot_axes['main'] = [ax0, ax1, ax2]
        self.plot_lines['main'] = [l0]

    # =========================================================================
    # Add fit window
    # =========================================================================

    def generateNewWindow(self):
        """."""
        # Run new window wizard
        dialog = NewWindowWizard(self)
        if not dialog.exec():
            return
        self.addFitWindow(**dialog.info)

    def addFitWindow(self, name):
        """."""
        # Check if the name exists
        if name in self.windows:
            logger.warning(f'{name} window already exists!')
            return

        # Generate the new tabs
        self.inputTabs[name] = QTabWidget()
        self.inputTabHolder.addTab(self.inputTabs[name], name)
        self.outputTabs[name] = QWidget()
        self.outputTabHolder.addTab(self.outputTabs[name], name)

        # Create widget holder
        winWidgets = Widgets()

        # Inputs tab ==========================================================

        setupTab = QScrollArea()
        gasTab = QScrollArea()
        shiftTab = QScrollArea()
        bgpolyTab = QScrollArea()
        self.inputTabs[name].addTab(setupTab, 'Setup')
        self.inputTabs[name].addTab(gasTab, 'Gases')
        self.inputTabs[name].addTab(shiftTab, 'Shift')
        self.inputTabs[name].addTab(bgpolyTab, 'Background')

        # Setup layout
        layout = QGridLayout(setupTab)

        # Create inputs for the fit window
        layout.addWidget(QRightLabel('Start\nWavenumber\n(cm-1)'), 0, 0)
        winWidgets['wn_start'] = SpinBox(0, [0, 1e5])
        layout.addWidget(winWidgets['wn_start'], 0, 1)
        layout.addWidget(QRightLabel('Stop\nWavenumber\n(cm-1)'), 0, 2)
        winWidgets['wn_stop'] = SpinBox(0, [0, 1e5])
        layout.addWidget(winWidgets['wn_stop'], 0, 3)

        # Input for apodization function
        layout.addWidget(QRightLabel('Apodization\nFunction'), 1, 0)
        winWidgets['apod_function'] = QComboBox()
        winWidgets['apod_function'].addItems(
            ['NB_weak', 'NB_medium', 'NB_strong', 'triangular', 'boxcar']
        )
        winWidgets.set('apod_function', 'NB_medium')
        layout.addWidget(winWidgets['apod_function'], 1, 1)

        # Input for fov
        layout.addWidget(QRightLabel('Field of View\n(radians)'), 1, 2)
        winWidgets['fov'] = DSpinBox(0.01, [0, 1], 0.01)
        layout.addWidget(winWidgets['fov'], 1, 3)
        winWidgets['fov_fit'] = QCheckBox('Fit?')
        winWidgets['fov_fit'].setChecked(True)
        layout.addWidget(winWidgets['fov_fit'], 1, 4)

        # Input for OPD
        layout.addWidget(QRightLabel('Optical Path\nDifference (cm)'), 2, 0)
        winWidgets['opd'] = DSpinBox(1.6, [0, 100], 0.01)
        layout.addWidget(winWidgets['opd'], 2, 1)

        # Input for Offset
        layout.addWidget(QRightLabel('Zero Offset (%)'), 2, 2)
        winWidgets['opd'] = DSpinBox(0, [0, 100], 1)
        layout.addWidget(winWidgets['opd'], 2, 3)

        # Add parameter tables
        winWidgets['gasTable'] = paramTable(gasTab, 'param', 450, 400,
                                            self.gas_list.keys())
        winWidgets['shiftTable'] = paramTable(shiftTab, 'poly')
        winWidgets['bgpolyTable'] = paramTable(bgpolyTab, 'poly')

        # Create a button to remove the window
        btn = QPushButton('Remove')
        btn.clicked.connect(lambda: self.remFitWindow(name))
        layout.addWidget(btn, 3, 3)

        # Outputs tab =========================================================

        # Setup layout
        layout = QGridLayout(self.outputTabs[name])

        # Generate the graph window
        graphwin = pg.GraphicsLayoutWidget(show=True)
        layout.addWidget(graphwin, 0, 0)

        # Generate the plot axes
        ax0 = graphwin.addPlot(row=0, col=0)
        ax1 = graphwin.addPlot(row=1, col=0)
        ax2 = graphwin.addPlot(row=2, col=0)

        # Greate the plot lines
        pen0 = pg.mkPen(color=self.PLOTCOLORS[0], width=1.0)
        pen1 = pg.mkPen(color=self.PLOTCOLORS[1], width=1.0)
        l0 = ax0.plot(pen=pen0)  # Measured spectrum
        l1 = ax0.plot(pen=pen1)  # Fitted spectrum
        l2 = ax1.plot(pen=pen0)  # residual
        l3 = ax2.plot(pen=pen0)  # Measured OD
        l4 = ax2.plot(pen=pen1)  # Fitted OD

        self.plot_axes[name] = [ax0, ax1, ax2]
        self.plot_lines[name] = [l0, l1, l2, l3, l4]

        # Add fit regions to main plot and connect to the wavenumber bounds
        self.plot_regions[name] = pg.LinearRegionItem([1000, 2000])
        self.plot_regions[name].setZValue(-1)
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

        # Add to overall widgets ==============================================

        self.windowWidgets[name] = winWidgets
        self.graphwins[name] = graphwin

        self.windows.append(name)

        logger.info(f'{name} window added')

    def remFitWindow(self, name):
        """Remove fit window."""
        # Get the index of the window tab
        window_idx = list(self.inputTabs.keys()).index(name) + 1

        # Remove the window tab from the GUI
        self.inputTabHolder.removeTab(window_idx)
        self.outputTabHolder.removeTab(window_idx)

        # Delete the actual widget from memory
        self.inputTabs[name].setParent(None)
        self.outputTabs[name].setParent(None)

        # Remove from list of windows
        self.windows.remove(name)

        # Remove window from main plot
        self.plot_axes['main'][0].removeItem(self.plot_regions[name])

        logger.info(f'{name} window removed')

    # =========================================================================
    # Analysis loop and slots
    # =========================================================================

    def begin_analysis(self):
        """Run main analysis loop."""
        # Pull the widget data
        widgetData = self.getWidgetData()
        self.workers = {}
        self.ready_flag = {}

        # Get the spectra to analyse
        self.spectra_list = widgetData['spec_fnames'].split('\n')
        self.spectrum_counter = 0

        self.update_status('Analysing')

        # Generate the thread workers
        for name in self.windows:
            worker = Worker(name, widgetData)
            worker.signals.results.connect(self.get_results)
            worker.signals.error.connect(self.update_error)
            worker.signals.finished.connect(self.analysis_complete)
            self.workers[name] = worker
            self.ready_flag[name] = False
            self.threadpool.start(worker)

        # Disable the start button and enable the pause/stop buttons
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def send_spectrum(self, spectrum):
        """Send spectrum to workers to analyse."""
        for name, worker in self.workers.items():
            worker.setSpectrum(spectrum)
            self.ready_flag[name] = False

    def stop_analysis(self):
        """Stop analysis."""
        for name, worker in self.workers.items():
            worker.stop()
        self.analysis_complete()

    def pause_analysis(self):
        """Pause the workers."""
        # Pause each worker
        for name, worker in self.workers.items():
            worker.pause()

        # Update button label
        if self.pause_btn.text() == 'Pause':
            self.pause_btn.setText('Continue')
        else:
            self.pause_btn.setText('Pause')

    @Slot(tuple)
    def get_results(self, results):
        """Catch results and flag that worker is ready for next spectrum."""
        # Unpack the results
        name, resdata = results

        # Update worker ready flag
        self.ready_flag[name] = True

        # Check if all workers are ready
        ready_flags = np.array([b for b in self.ready_flag.values()])
        if ready_flags.all():

            # If so, check if all spectra have been analysed
            if self.spectrum_counter == len(self.spectra_list):
                self.stop_analysis()

            # If not, analyse the next spectrum
            else:
                self.spec_filename = self.spectra_list[self.spectrum_counter]
                self.spectrum = read_spectrum(self.spec_filename)
                self.send_spectrum(self.spectrum)
                self.spectrum_counter += 1
                self.update_main_plots()

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

    def update_main_plots(self):
        """."""
        # Get the spectrum x and y data
        x = self.spectrum.coords['Wavenumber'].to_numpy()
        y = self.spectrum.to_numpy()

        self.plot_lines['main'][0].setData(x, y)

    def update_window_plots(self, plotData):
        """Update output plots."""

    def update_error(self, error):
        """Update error messages from the worker."""
        exctype, value, trace = error
        logger.warning(f'Uncaught exception!\n{trace}')

    def update_status(self, status):
        """Update status bar."""
        self.statusBar().showMessage(status)

    def update_progress(self):
        """Update the progress bar."""

    # =========================================================================
    # Program Global Slots
    # =========================================================================

    @Slot(str, logging.LogRecord)
    def updateLog(self, status, record):
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
            if cwd in fname:
                fname = fname[len(cwd):]
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
                if key in ['gasTable', 'shiftTable', 'bgpolyTable']:
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
                if key in ['gasTable', 'shiftTable', 'bgpolyTable']:
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
            fname, tfile = QFileDialog.getOpenFileName(self, 'Load Config', '',
                                                       filter)

        # Set param table keys
        tableKeys = ['gasTable', 'shiftTable', 'bgpolyTable']

        # Open the config file
        try:
            with open(fname, 'r') as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.FullLoader)

            # Apply each config setting
            for label, value in config.items():
                # try:
                # Set the fit windows
                if label == 'fitWindows':
                    for name, widgets in value.items():

                        # Generate the window tabs
                        self.addFitWindow(name)

                        for key, val in widgets.items():

                            # Setup the parameter tables
                            if key in tableKeys:
                                self.windowWidgets[name][key].setData(val)
                            else:
                                self.windowWidgets[name].set(key, val)

                elif label == 'theme':
                    self.theme = value

                else:
                    self.widgets.set(label, value)
                # except Exception:
                #     logger.warning(f'Failed to load {label} from config file')

            logger.info(f'Config file loaded from {self.config_fname}')

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
        """Change the theme."""
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

        # Set graph background color
        for graphwin in self.graphwins.values():
            graphwin.setBackground(bg_color)

        # Set axes spines color
        for axes in self.plot_axes.values():
            for ax in axes:
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
    """."""
    results = Signal(tuple)
    finished = Signal()
    error = Signal(tuple)


class Worker(QRunnable):
    """."""

    def __init__(self, name, widgetData, *args, **kwargs):
        """."""
        super(Worker, self).__init__()
        self.name = name
        self.widgetData = widgetData
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.isStopped = False
        self.isPaused = False
        self.spectrum = None

    @Slot()
    def run(self):
        """."""
        try:
            self.fn(self.name, *self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        logger.info(f'{self.name} analyser finished')

    def fn(self, name):
        """Main analysis loop."""
        time.sleep(2.0)
        self.signals.results.emit([name, None])
        while not self.isStopped:
            if self.spectrum is not None and not self.isPaused:
                resdata = {}
                time.sleep(1.0)
                self.spectrum = None
                self.signals.results.emit([name, resdata])
            else:
                time.sleep(0.01)

    def setSpectrum(self, spectrum):
        """Set new spectrum to analyse."""
        self.spectrum = spectrum

    def pause(self):
        """Pause the analysis."""
        if self.isPaused:
            self.isPaused = False
        else:
            self.isPaused = True

    def stop(self):
        """Stop the analysis."""
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
            raise ValueError

    def set(self, key, value):
        """Set the value of a widget."""
        if type(self[key]) in [QTextEdit, QLineEdit]:
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
            raise ValueError


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
                self.setItem(i, 3, QTableWidgetItem(line[3]))
                self.setItem(i, 4, QTableWidgetItem(line[4]))
                self.setItem(i, 5, QTableWidgetItem(line[5]))

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
        """."""
        self.setColumnWidth(0, 80)
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
# =============================================================================
# Client code
# =============================================================================
# =============================================================================

def main():
    QThread.currentThread().setObjectName('MainThread')
    logging.getLogger().setLevel(logging.INFO)
    app = QApplication(sys.argv)
    window = MainWindow(app)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
