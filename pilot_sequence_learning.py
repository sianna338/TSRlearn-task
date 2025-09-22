#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on September 21, 2025, at 11:57
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2024.2.4')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard
from psychopy_bids.bids import BIDSHandler

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'pilot_sequence_learning'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '0',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\sync_folder\\TSRlearn-task\\pilot_sequence_learning.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[-0.7804, -0.7804, -0.7804], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=False,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-0.7804, -0.7804, -0.7804]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('continue_button') is None:
        # initialise continue_button
        continue_button = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue_button',
        )
    if deviceManager.getDevice('continue_button_2') is None:
        # initialise continue_button_2
        continue_button_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue_button_2',
        )
    if deviceManager.getDevice('continue_button_4') is None:
        # initialise continue_button_4
        continue_button_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue_button_4',
        )
    if deviceManager.getDevice('key_resp_prc') is None:
        # initialise key_resp_prc
        key_resp_prc = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_prc',
        )
    if deviceManager.getDevice('continue_button_5') is None:
        # initialise continue_button_5
        continue_button_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue_button_5',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('continue_button_3') is None:
        # initialise continue_button_3
        continue_button_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue_button_3',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "startup_settings" ---
    # Run 'Begin Experiment' code from functions_imports
    
    # log function that saves variables to log file
    def log(*msgs, sep=' ', end='\n'):
        log_file = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'] + '_print.log')
        print(*msgs, sep=sep, end=end, flush=True)
    
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(sep.join(str(msg) for msg in msgs) + end)
            
     # import meg_trigger package
    import meg_triggers
    from meg_triggers import send_trigger
    meg_triggers.enable_printing()
    meg_triggers.set_default_duration(0.005)
    
    # function for setting triggers when comps appear
    def send_onset_trigger(stim, trig_number):
        # record exact onset on the global clock 
        win.timeOnFlip(stim, 'tStartRefresh')
        log(f'{stim}.started')
        send_trigger(trig_number)
        
    from ast import literal_eval
    
    # define function that matches position keywords to tuples
    def resolve_pos(val):
        """Return a (x,y) tuple from various spreadsheet formats."""
        # Already a tuple/list/ndarray -> normalize to tuple
        if isinstance(val, (list, tuple, np.ndarray)):
            return tuple(val)
        # Name like "left_pos" -> look up
        if isinstance(val, str):
            name = val.strip()
            if name in POS:
                return POS[name]
            # Or literal like "(-0.4, 0.2)" in the cell
            try:
                parsed = literal_eval(name)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                    return tuple(parsed)
            except Exception:
                pass
            raise ValueError(f"Unrecognized position value: {val!r}")
        raise TypeError(f"Unsupported position type: {type(val)}")
    
    # Run 'Begin Experiment' code from exp_settings
    
    ## set main experimental variables
    
    # instruction language
    language = "english" # "german" or "english"
    
    # keys
    left_key = "left"
    center_key = "up"
    right_key = "right" 
    
    # pos of images (x-coord, y-coord)
    left_pos   = (-0.3, 0.15)
    right_pos  = (0.3, 0.15)
    center_pos = (0.0, 0.15)
    prompt_pos = (0, -0.15) 
    
    # # Map positions to keys
    pos_to_key = {
        right_pos: right_key,
        center_pos: center_key,
        left_pos: left_key,
    }
    # pos of instructions
    instruc_pos = (0,0)
    
    # make coordinates her match with position description in excel file 
    POS = {
        'left':   left_pos,
        'center': center_pos,
        'right':  right_pos,
    }
     
     # timings
    max_response_time = 10 # [s]
     
     # feedback
    feedback_steps  =  0.04   # steps out of 1
    rest_jump       =  0.004 # abs distance to target pos
    animation_time   = 2.5    # [s]
     
    # breaks (indicate after whch trial number you want to break)
    break_points = {60, 120, 180}
    break_dur = 30
    # Run 'Begin Experiment' code from bids_logging_functions
    
    # --- BIDS event logger (no Builder BIDS components needed) ---
    from pathlib import Path
    import csv
    import numpy as np
    import atexit
    
    class BIDSLogger:
        def __init__(self, win, clock, default_cols=None):
            self.win = win
            self.clock = clock
            self.rows = []          # list of dicts -> will become events.tsv
            self.active = {}        # stim -> row index (for duration)
            self.defaults = default_cols or {}
    
            # fixed column order (add any you want to see in the TSV)
            self.col_order = [
                "subject", "block_name", "sequence_name", 
                "route_num",
                "trial_num", "sequence_name", 
                "component_label", 
                "concept_label", "concept_exemplar", 
                "onset", "duration",
                "type_of_stimulus",
                "response_time", "response",
                "response_meaning", "expected_response",
                "correct"
            ]
    
        # schedule logging of onset on the *next* flip (exactly when the stim appears)
        def schedule_onset(self, stim, **extra_cols):
             # Onset: first frame it is STARTED
            if getattr(stim, "status", None) == STARTED and stim not in self.active:
                # Builder already records exact onset on flip into tStartRefresh;
                # fall back to current clock if that isn't available.
                t_on = getattr(stim, "tStartRefresh", None)
                if t_on is None:
                    t_on = self.clock.getTime()
                row = {
                    "onset": float(t_on),
                    "duration": np.nan,
                }
                row.update(self.defaults)
                row.update(extra_cols)
                self.rows.append(row)
                self.active[stim] = len(self.rows) - 1
        
        # to log the offset and therefore compute durations of components
        def mark_offset(self, stim):
            if getattr(stim, "status", None) == FINISHED and stim in self.active:
                idx = self.active.pop(stim)
                t_off = getattr(stim, "tStopRefresh", None)
                if t_off is None:
                    t_off = self.clock.getTime()
                self.rows[idx]["duration"] = float(t_off - self.rows[idx]["onset"])
                
        # log a one-shot event right now (no duration), e.g., button press
        def add_instant(self, trial_type, **extra_cols):
            row = {"onset": float(self.clock.getTime()), "duration": 0.0,
                   "type_of_stimulus": trial_type}
            row.update(self.defaults)
            row.update(extra_cols)
            self.rows.append(row)
    
        # write out a BIDS-like events.tsv
        def save(self, filename_base):
            out = Path(filename_base).with_suffix("")  # strip .csv/.log if present
            out = out.parent / (out.name + "_events.tsv")
    
            # union of all keys, but keep preferred order first
            all_keys = list(self.col_order)
            for r in self.rows:
                for k in r.keys():
                    if k not in all_keys:
                        all_keys.append(k)
    
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=all_keys, delimiter="\t", extrasaction="ignore")
                w.writeheader()
                for r in self.rows:
                    # convert Nones to "n/a" for BIDS friendliness
                    out_row = {k: ("n/a" if (r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k)))) else r.get(k))
                               for k in all_keys}
                    w.writerow(out_row)
    
            return str(out)
    # set global clock as default because we need it for timing
    bids = None
    
    # atexit autosave: runs on normal finish AND on ESC (core.quit)
    def _bids_autosave():
        try:
            if bids and hasattr(bids, "rows"):
                out = bids.save(thisExp.dataFileName)
                print(f"[autosave] BIDS events written to: {out}")
        except Exception as e:
            print(f"[autosave] Failed to save BIDS events: {e}")
    
    
    
    # --- Initialize components for Routine "instructions" ---
    # Run 'Begin Experiment' code from trigger_start
    # set trigger that exp has started
    send_trigger(255)
    
    instruction_part1 = visual.TextStim(win=win, name='instruction_part1',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # Run 'Begin Experiment' code from instruction_part1_text
    
    
    continue_button = keyboard.Keyboard(deviceName='continue_button')
    
    # --- Initialize components for Routine "instructions_02" ---
    instruction_part2 = visual.TextStim(win=win, name='instruction_part2',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from instructions_part2_text
    
    
    continue_button_2 = keyboard.Keyboard(deviceName='continue_button_2')
    
    # --- Initialize components for Routine "instructions_03" ---
    instruction_part3 = visual.TextStim(win=win, name='instruction_part3',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from instruction_part3_text
    
    
    continue_button_4 = keyboard.Keyboard(deviceName='continue_button_4')
    
    # --- Initialize components for Routine "choice_disp_learn" ---
    prompt_prc = visual.ImageStim(
        win=win,
        name='prompt_prc', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    dist_01_prc = visual.ImageStim(
        win=win,
        name='dist_01_prc', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    dist_02_prc = visual.ImageStim(
        win=win,
        name='dist_02_prc', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    correct_prc = visual.ImageStim(
        win=win,
        name='correct_prc', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    key_resp_prc = keyboard.Keyboard(deviceName='key_resp_prc')
    instructions_choose_prc = visual.TextStim(win=win, name='instructions_choose_prc',
        text=None,
        font='Arial',
        pos=instruc_pos, draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "feedback_learn" ---
    # Run 'Begin Experiment' code from animation_control_prc
    
    
    prompt_prc_2 = visual.ImageStim(
        win=win,
        name='prompt_prc_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    dist_01_prc_2 = visual.ImageStim(
        win=win,
        name='dist_01_prc_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    dist_02_prc_2 = visual.ImageStim(
        win=win,
        name='dist_02_prc_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    correct_prc_2 = visual.ImageStim(
        win=win,
        name='correct_prc_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2,0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "instructions_start" ---
    instruction_part4 = visual.TextStim(win=win, name='instruction_part4',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from instruction_part4_text
    
    
    continue_button_5 = keyboard.Keyboard(deviceName='continue_button_5')
    
    # --- Initialize components for Routine "set_tracking_parameters" ---
    # Run 'Begin Experiment' code from set_parameters
    # counters for how many routes of each sequence have started so far
    A_route = 0
    B_route = 0
    C_route = 0
    # Run 'Begin Experiment' code from log_trial_numbers
    
    
    
    # --- Initialize components for Routine "choice_display" ---
    prompt = visual.ImageStim(
        win=win,
        name='prompt', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    dist_01 = visual.ImageStim(
        win=win,
        name='dist_01', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    dist_02 = visual.ImageStim(
        win=win,
        name='dist_02', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    correct = visual.ImageStim(
        win=win,
        name='correct', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    instructions_choose = visual.TextStim(win=win, name='instructions_choose',
        text=None,
        font='Arial',
        pos=instruc_pos, draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from animation_control
    
    
    prompt_2 = visual.ImageStim(
        win=win,
        name='prompt_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    dist_01_2 = visual.ImageStim(
        win=win,
        name='dist_01_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    dist_02_2 = visual.ImageStim(
        win=win,
        name='dist_02_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    correct_2 = visual.ImageStim(
        win=win,
        name='correct_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    
    # --- Initialize components for Routine "task_break" ---
    breaks_instruction = visual.TextStim(win=win, name='breaks_instruction',
        text='break, the task will continue soon',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instructions_end" ---
    instruction_end = visual.TextStim(win=win, name='instruction_end',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from instructions_end_text
    
    
    continue_button_3 = keyboard.Keyboard(deviceName='continue_button_3')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "startup_settings" ---
    # create an object to store info about Routine startup_settings
    startup_settings = data.Routine(
        name='startup_settings',
        components=[],
    )
    startup_settings.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from bids_logging_functions
    # ---- Create the logger once ----
    bids = BIDSLogger(
        win=win,
        clock=globalClock,
        default_cols=dict(
            subject=expInfo.get("participant")
        )
    )
    
    # write a header-only file immediately so each start creates a file
    _ = bids.save(thisExp.dataFileName)
    
    # atexit autosave: runs on normal finish AND on ESC (core.quit)
    atexit.register(_bids_autosave)
    
    # store start times for startup_settings
    startup_settings.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    startup_settings.tStart = globalClock.getTime(format='float')
    startup_settings.status = STARTED
    thisExp.addData('startup_settings.started', startup_settings.tStart)
    startup_settings.maxDuration = None
    # keep track of which components have finished
    startup_settingsComponents = startup_settings.components
    for thisComponent in startup_settings.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "startup_settings" ---
    startup_settings.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            startup_settings.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in startup_settings.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "startup_settings" ---
    for thisComponent in startup_settings.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for startup_settings
    startup_settings.tStop = globalClock.getTime(format='float')
    startup_settings.tStopRefresh = tThisFlipGlobal
    thisExp.addData('startup_settings.stopped', startup_settings.tStop)
    thisExp.nextEntry()
    # the Routine "startup_settings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[instruction_part1, continue_button],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from instruction_part1_text
    if language == "english": 
        instruction_part1.text = (
        "Welcome to the second part of the experiment. "
        "You will now learn sequences of images by moving through them, one pair at a time. "
        "Press any button to continue."
    )
    
       
    if language == "german": 
        instruction_part1.text = ("Willkommen zum zweiten Teil des Experiments. "
        "Sie werden nun Bildsequenzen lernen. "
        "Sie werden die Sequenzen erleben, indem Sie sie durchlaufen und von einem Bild zum nächsten springen. "
        "Drücken Sie irgendeinen Knopf, um weiterzumachen. ")
    
    # create starting attributes for continue_button
    continue_button.keys = []
    continue_button.rt = []
    _continue_button_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from bids_instruc
        bids.schedule_onset(instruction_part1,
                                type_of_stimulus="instructions",
                                component_label="instruction_part1")
        
        # *instruction_part1* updates
        
        # if instruction_part1 is starting this frame...
        if instruction_part1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_part1.frameNStart = frameN  # exact frame index
            instruction_part1.tStart = t  # local t and not account for scr refresh
            instruction_part1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_part1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_part1.started')
            # update status
            instruction_part1.status = STARTED
            instruction_part1.setAutoDraw(True)
        
        # if instruction_part1 is active this frame...
        if instruction_part1.status == STARTED:
            # update params
            pass
        
        # *continue_button* updates
        waitOnFlip = False
        
        # if continue_button is starting this frame...
        if continue_button.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_button.frameNStart = frameN  # exact frame index
            continue_button.tStart = t  # local t and not account for scr refresh
            continue_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_button.started')
            # update status
            continue_button.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_button.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_button.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_button.status == STARTED and not waitOnFlip:
            theseKeys = continue_button.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
            _continue_button_allKeys.extend(theseKeys)
            if len(_continue_button_allKeys):
                continue_button.keys = _continue_button_allKeys[-1].name  # just the last key pressed
                continue_button.rt = _continue_button_allKeys[-1].rt
                continue_button.duration = _continue_button_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # Run 'End Routine' code from bids_instruc
    bids.mark_offset(instruction_part1)
    # check responses
    if continue_button.keys in ['', [], None]:  # No response was made
        continue_button.keys = None
    thisExp.addData('continue_button.keys',continue_button.keys)
    if continue_button.keys != None:  # we had a response
        thisExp.addData('continue_button.rt', continue_button.rt)
        thisExp.addData('continue_button.duration', continue_button.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_02" ---
    # create an object to store info about Routine instructions_02
    instructions_02 = data.Routine(
        name='instructions_02',
        components=[instruction_part2, continue_button_2],
    )
    instructions_02.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from instructions_part2_text
    if language == "english": 
        instruction_part2.text = (
        "Starting at the first image, choose the next image in the sequence by trial and error. "
        "After each choice, you’ll get feedback: the correct next image will move down and replace the current one. "
        "Then you choose again. "
        "Press any button to continue. "
     
        )
    
       
    if language == "german": 
        instruction_part2.text = (
        "Sie beginnen nun am Anfang der ersten Sequenz und sollten durch Ausprobieren das nächste Bild in der Sequenz auswählen. " 
        "Sie erhalten Feedback zu Ihrer Auswahl, da das richtige nächste Bild das erste Bild ersetzt. "
        "Dann treffen Sie die nächste Auswahl. "
        "Drücken Sie irgendeinen Knopf, um weiterzumachen."
        )
    
    
    # create starting attributes for continue_button_2
    continue_button_2.keys = []
    continue_button_2.rt = []
    _continue_button_2_allKeys = []
    # store start times for instructions_02
    instructions_02.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_02.tStart = globalClock.getTime(format='float')
    instructions_02.status = STARTED
    thisExp.addData('instructions_02.started', instructions_02.tStart)
    instructions_02.maxDuration = None
    # keep track of which components have finished
    instructions_02Components = instructions_02.components
    for thisComponent in instructions_02.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_02" ---
    instructions_02.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from bids_instruc_2
        bids.schedule_onset(instruction_part2,
                                type_of_stimulus="instructions",
                                component_label="instruction_part2")
        
        # *instruction_part2* updates
        
        # if instruction_part2 is starting this frame...
        if instruction_part2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_part2.frameNStart = frameN  # exact frame index
            instruction_part2.tStart = t  # local t and not account for scr refresh
            instruction_part2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_part2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_part2.started')
            # update status
            instruction_part2.status = STARTED
            instruction_part2.setAutoDraw(True)
        
        # if instruction_part2 is active this frame...
        if instruction_part2.status == STARTED:
            # update params
            pass
        
        # *continue_button_2* updates
        waitOnFlip = False
        
        # if continue_button_2 is starting this frame...
        if continue_button_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_button_2.frameNStart = frameN  # exact frame index
            continue_button_2.tStart = t  # local t and not account for scr refresh
            continue_button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_button_2.started')
            # update status
            continue_button_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_button_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_button_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_button_2.status == STARTED and not waitOnFlip:
            theseKeys = continue_button_2.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
            _continue_button_2_allKeys.extend(theseKeys)
            if len(_continue_button_2_allKeys):
                continue_button_2.keys = _continue_button_2_allKeys[-1].name  # just the last key pressed
                continue_button_2.rt = _continue_button_2_allKeys[-1].rt
                continue_button_2.duration = _continue_button_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_02.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_02.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_02" ---
    for thisComponent in instructions_02.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_02
    instructions_02.tStop = globalClock.getTime(format='float')
    instructions_02.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_02.stopped', instructions_02.tStop)
    # Run 'End Routine' code from bids_instruc_2
    bids.mark_offset(instruction_part2) 
    # check responses
    if continue_button_2.keys in ['', [], None]:  # No response was made
        continue_button_2.keys = None
    thisExp.addData('continue_button_2.keys',continue_button_2.keys)
    if continue_button_2.keys != None:  # we had a response
        thisExp.addData('continue_button_2.rt', continue_button_2.rt)
        thisExp.addData('continue_button_2.duration', continue_button_2.duration)
    thisExp.nextEntry()
    # the Routine "instructions_02" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_03" ---
    # create an object to store info about Routine instructions_03
    instructions_03 = data.Routine(
        name='instructions_03',
        components=[instruction_part3, continue_button_4],
    )
    instructions_03.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from instruction_part3_text
    if language == "english": 
        instruction_part3.text = (
        "We will start with a few practice trials. "
        "The current image appears at the bottom of the screen, and three options appear at the top. "
        "Choose the image you think comes next in the sequence. "
        "Press any key to start the practice trials."
        )
    
        
        
    if language == "german": 
        instruction_part3.text = (
            "Wir beginnen mit einigen Übungsdurchgängen. "
            "Unten auf dem Bildschirm sehen Sie das aktuelle Bild, oben drei Auswahlmöglichkeiten. "
            "Wählen Sie das Bild, von dem Sie denken, dass es als Nächstes in der Abfolge kommt. "
            "Drücken Sie irgendeinen Knopf, um mit den Übungsdurchgängen zu beginnen."
        )
    
    
    # create starting attributes for continue_button_4
    continue_button_4.keys = []
    continue_button_4.rt = []
    _continue_button_4_allKeys = []
    # store start times for instructions_03
    instructions_03.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_03.tStart = globalClock.getTime(format='float')
    instructions_03.status = STARTED
    thisExp.addData('instructions_03.started', instructions_03.tStart)
    instructions_03.maxDuration = None
    # keep track of which components have finished
    instructions_03Components = instructions_03.components
    for thisComponent in instructions_03.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_03" ---
    instructions_03.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from bids_instruc_3
        bids.schedule_onset(instruction_part3,
                                type_of_stimulus="instructions",
                                component_label="instruction_part2")
        
        # *instruction_part3* updates
        
        # if instruction_part3 is starting this frame...
        if instruction_part3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_part3.frameNStart = frameN  # exact frame index
            instruction_part3.tStart = t  # local t and not account for scr refresh
            instruction_part3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_part3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_part3.started')
            # update status
            instruction_part3.status = STARTED
            instruction_part3.setAutoDraw(True)
        
        # if instruction_part3 is active this frame...
        if instruction_part3.status == STARTED:
            # update params
            pass
        
        # *continue_button_4* updates
        waitOnFlip = False
        
        # if continue_button_4 is starting this frame...
        if continue_button_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_button_4.frameNStart = frameN  # exact frame index
            continue_button_4.tStart = t  # local t and not account for scr refresh
            continue_button_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_button_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_button_4.started')
            # update status
            continue_button_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_button_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_button_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_button_4.status == STARTED and not waitOnFlip:
            theseKeys = continue_button_4.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
            _continue_button_4_allKeys.extend(theseKeys)
            if len(_continue_button_4_allKeys):
                continue_button_4.keys = _continue_button_4_allKeys[-1].name  # just the last key pressed
                continue_button_4.rt = _continue_button_4_allKeys[-1].rt
                continue_button_4.duration = _continue_button_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_03.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_03.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_03" ---
    for thisComponent in instructions_03.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_03
    instructions_03.tStop = globalClock.getTime(format='float')
    instructions_03.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_03.stopped', instructions_03.tStop)
    # Run 'End Routine' code from bids_instruc_3
    bids.mark_offset(instruction_part3) 
    # check responses
    if continue_button_4.keys in ['', [], None]:  # No response was made
        continue_button_4.keys = None
    thisExp.addData('continue_button_4.keys',continue_button_4.keys)
    if continue_button_4.keys != None:  # we had a response
        thisExp.addData('continue_button_4.rt', continue_button_4.rt)
        thisExp.addData('continue_button_4.duration', continue_button_4.duration)
    thisExp.nextEntry()
    # the Routine "instructions_03" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_trials = data.TrialHandler2(
        name='practice_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('sequences/main_trials_prc.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(practice_trials)  # add the loop to the experiment
    thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
    if thisPractice_trial != None:
        for paramName in thisPractice_trial:
            globals()[paramName] = thisPractice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_trial in practice_trials:
        currentLoop = practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        # --- Prepare to start Routine "choice_disp_learn" ---
        # create an object to store info about Routine choice_disp_learn
        choice_disp_learn = data.Routine(
            name='choice_disp_learn',
            components=[prompt_prc, dist_01_prc, dist_02_prc, correct_prc, key_resp_prc, instructions_choose_prc],
        )
        choice_disp_learn.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prompt_prc.setPos(prompt_pos)
        prompt_prc.setImage('stimuli\\practice_images\\' + promptFile)
        dist_01_prc.setPos([resolve_pos(dist01_pos)])
        dist_01_prc.setImage('stimuli\\practice_images\\' + dist_01File)
        dist_02_prc.setPos([resolve_pos(dist02_pos)])
        dist_02_prc.setImage('stimuli\\practice_images\\' + dist_02File)
        correct_prc.setPos([resolve_pos(correct_pos)])
        correct_prc.setImage('stimuli\\practice_images\\' + correctFile)
        # create starting attributes for key_resp_prc
        key_resp_prc.keys = []
        key_resp_prc.rt = []
        _key_resp_prc_allKeys = []
        # Run 'Begin Routine' code from code_prc
        # initialize
        responded = False
        key_resp_prc.keys = []
        key_resp_prc.rt = None
        
        # this attribute needs to also exist otherwise there is error
        key_resp_prc.duration = None
        
        # start a clock to time out routine if necessary
        routClock = core.Clock()
        routClock.reset()
        
        
        # Run 'Begin Routine' code from instructions_choose_txt_prc
        if language == "english":
            instructions_choose_prc.text = (
            "Choose the next image in the sequence!")
         
        if language == "german":
            instructions_choose_prc.text = (
            "Wähle das nächste Bild in der Sequenz!")
        # store start times for choice_disp_learn
        choice_disp_learn.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        choice_disp_learn.tStart = globalClock.getTime(format='float')
        choice_disp_learn.status = STARTED
        thisExp.addData('choice_disp_learn.started', choice_disp_learn.tStart)
        choice_disp_learn.maxDuration = None
        # keep track of which components have finished
        choice_disp_learnComponents = choice_disp_learn.components
        for thisComponent in choice_disp_learn.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "choice_disp_learn" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        choice_disp_learn.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prompt_prc* updates
            
            # if prompt_prc is starting this frame...
            if prompt_prc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompt_prc.frameNStart = frameN  # exact frame index
                prompt_prc.tStart = t  # local t and not account for scr refresh
                prompt_prc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompt_prc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_prc.started')
                # update status
                prompt_prc.status = STARTED
                prompt_prc.setAutoDraw(True)
            
            # if prompt_prc is active this frame...
            if prompt_prc.status == STARTED:
                # update params
                pass
            
            # *dist_01_prc* updates
            
            # if dist_01_prc is starting this frame...
            if dist_01_prc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_01_prc.frameNStart = frameN  # exact frame index
                dist_01_prc.tStart = t  # local t and not account for scr refresh
                dist_01_prc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_01_prc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_01_prc.started')
                # update status
                dist_01_prc.status = STARTED
                dist_01_prc.setAutoDraw(True)
            
            # if dist_01_prc is active this frame...
            if dist_01_prc.status == STARTED:
                # update params
                pass
            
            # *dist_02_prc* updates
            
            # if dist_02_prc is starting this frame...
            if dist_02_prc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_02_prc.frameNStart = frameN  # exact frame index
                dist_02_prc.tStart = t  # local t and not account for scr refresh
                dist_02_prc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_02_prc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_02_prc.started')
                # update status
                dist_02_prc.status = STARTED
                dist_02_prc.setAutoDraw(True)
            
            # if dist_02_prc is active this frame...
            if dist_02_prc.status == STARTED:
                # update params
                pass
            
            # *correct_prc* updates
            
            # if correct_prc is starting this frame...
            if correct_prc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                correct_prc.frameNStart = frameN  # exact frame index
                correct_prc.tStart = t  # local t and not account for scr refresh
                correct_prc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correct_prc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'correct_prc.started')
                # update status
                correct_prc.status = STARTED
                correct_prc.setAutoDraw(True)
            
            # if correct_prc is active this frame...
            if correct_prc.status == STARTED:
                # update params
                pass
            
            # *key_resp_prc* updates
            
            # if key_resp_prc is starting this frame...
            if key_resp_prc.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_prc.frameNStart = frameN  # exact frame index
                key_resp_prc.tStart = t  # local t and not account for scr refresh
                key_resp_prc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_prc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('key_resp_prc.started', t)
                # update status
                key_resp_prc.status = STARTED
                # keyboard checking is just starting
                key_resp_prc.clock.reset()  # now t=0
                key_resp_prc.clearEvents(eventType='keyboard')
            
            # if key_resp_prc is stopping this frame...
            if key_resp_prc.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_prc.tStartRefresh + max_response_time-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_prc.tStop = t  # not accounting for scr refresh
                    key_resp_prc.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_prc.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('key_resp_prc.stopped', t)
                    # update status
                    key_resp_prc.status = FINISHED
                    key_resp_prc.status = FINISHED
            if key_resp_prc.status == STARTED:
                theseKeys = key_resp_prc.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
                _key_resp_prc_allKeys.extend(theseKeys)
                if len(_key_resp_prc_allKeys):
                    key_resp_prc.keys = _key_resp_prc_allKeys[0].name  # just the first key pressed
                    key_resp_prc.rt = _key_resp_prc_allKeys[0].rt
                    key_resp_prc.duration = _key_resp_prc_allKeys[0].duration
                    # was this correct?
                    if (key_resp_prc.keys == str(correct_ans)) or (key_resp_prc.keys == correct_ans):
                        key_resp_prc.corr = 1
                    else:
                        key_resp_prc.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_prc
            # timeout case
            if routClock.getTime() >= max_response_time:
                # no key was pressed in time
                key_resp_prc.keys = None
                key_resp_prc.rt = None
                responded = False
                continueRoutine = False 
                # (optionally store that it was a timeout)
                #thisExp.addData('timeout', True)
                
            # if there is key selection 
            if key_resp_prc.status == STARTED:
                theseKeys = key_resp_prc.getKeys(keyList=['y','g','b','r'], ignoreKeys=["escape"], waitRelease=True)
                if len(theseKeys):
                    key_resp_prc.keys     = theseKeys[0].name
                    key_resp_prc.rt       = theseKeys[0].rt
                    key_resp_prc.duration = theseKeys[0].duration  # safe to save later
                    key_resp_prc.corr = 1 if key_resp_prc.keys == correctAnsKey else 0
                    responded = True
                    continueRoutine = False
            
            # *instructions_choose_prc* updates
            
            # if instructions_choose_prc is starting this frame...
            if instructions_choose_prc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instructions_choose_prc.frameNStart = frameN  # exact frame index
                instructions_choose_prc.tStart = t  # local t and not account for scr refresh
                instructions_choose_prc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instructions_choose_prc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions_choose_prc.started')
                # update status
                instructions_choose_prc.status = STARTED
                instructions_choose_prc.setAutoDraw(True)
            
            # if instructions_choose_prc is active this frame...
            if instructions_choose_prc.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                choice_disp_learn.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in choice_disp_learn.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "choice_disp_learn" ---
        for thisComponent in choice_disp_learn.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for choice_disp_learn
        choice_disp_learn.tStop = globalClock.getTime(format='float')
        choice_disp_learn.tStopRefresh = tThisFlipGlobal
        thisExp.addData('choice_disp_learn.stopped', choice_disp_learn.tStop)
        # check responses
        if key_resp_prc.keys in ['', [], None]:  # No response was made
            key_resp_prc.keys = None
            # was no response the correct answer?!
            if str(correct_ans).lower() == 'none':
               key_resp_prc.corr = 1;  # correct non-response
            else:
               key_resp_prc.corr = 0;  # failed to respond (incorrectly)
        # store data for practice_trials (TrialHandler)
        practice_trials.addData('key_resp_prc.keys',key_resp_prc.keys)
        practice_trials.addData('key_resp_prc.corr', key_resp_prc.corr)
        if key_resp_prc.keys != None:  # we had a response
            practice_trials.addData('key_resp_prc.rt', key_resp_prc.rt)
            practice_trials.addData('key_resp_prc.duration', key_resp_prc.duration)
        # the Routine "choice_disp_learn" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback_learn" ---
        # create an object to store info about Routine feedback_learn
        feedback_learn = data.Routine(
            name='feedback_learn',
            components=[prompt_prc_2, dist_01_prc_2, dist_02_prc_2, correct_prc_2],
        )
        feedback_learn.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from animation_control_prc
        
        # Initialize animation control
        endY = prompt_pos[1]    # end y-coor of moving image
        endX = prompt_pos[0]    # end x-coord of moving image
        
        maxAnimationDur = int(round(animation_time * expInfo['frameRate']))
        animationTimer = 0      # initialize variable 
        animationDone = False   # initialize variable 
        moveCorrect = False     # initialize variable 
        prompt_prc_2.setPos(prompt_pos)
        prompt_prc_2.setImage('stimuli\\practice_images\\' + promptFile)
        dist_01_prc_2.setPos([resolve_pos(dist01_pos)])
        dist_01_prc_2.setImage('stimuli\\practice_images\\' + dist_01File)
        dist_02_prc_2.setPos([resolve_pos(dist02_pos)])
        dist_02_prc_2.setImage('stimuli\\practice_images\\' + dist_02File)
        correct_prc_2.setPos([resolve_pos(correct_pos)])
        correct_prc_2.setImage('stimuli\\practice_images\\' + correctFile)
        # store start times for feedback_learn
        feedback_learn.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_learn.tStart = globalClock.getTime(format='float')
        feedback_learn.status = STARTED
        thisExp.addData('feedback_learn.started', feedback_learn.tStart)
        feedback_learn.maxDuration = None
        # keep track of which components have finished
        feedback_learnComponents = feedback_learn.components
        for thisComponent in feedback_learn.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_learn" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        feedback_learn.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from animation_control_prc
            # Start animation 
            if not moveCorrect and not animationDone:
                moveCorrect = True 
            
            # Run animation
            if moveCorrect and not animationDone:
                animationTimer += 1
                
                # Current position
                current_x, current_y = correct_prc_2.pos
                
                # Compute direction toward target
                dx = endX - current_x
                dy = endY - current_y
            
                # Move a small fraction toward target
                move_fraction = feedback_steps # % of the remaining distance each frame
                new_x = current_x + dx * move_fraction
                new_y = current_y + dy * move_fraction
            
                # Update position
                correct_prc_2.setPos((new_x, new_y))
            
                # Stop when close enough to target
                if (abs(dx) < rest_jump and abs(dy) < rest_jump) or animationTimer > maxAnimationDur:
                # if (abs(dx) < rest_jump and abs(dy) < rest_jump):
                    correct_prc_2.setPos((endX, endY))
                    animationDone = True
                    moveCorrect = False
                    continueRoutine = False
            
            # *prompt_prc_2* updates
            
            # if prompt_prc_2 is starting this frame...
            if prompt_prc_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompt_prc_2.frameNStart = frameN  # exact frame index
                prompt_prc_2.tStart = t  # local t and not account for scr refresh
                prompt_prc_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompt_prc_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_prc_2.started')
                # update status
                prompt_prc_2.status = STARTED
                prompt_prc_2.setAutoDraw(True)
            
            # if prompt_prc_2 is active this frame...
            if prompt_prc_2.status == STARTED:
                # update params
                pass
            
            # *dist_01_prc_2* updates
            
            # if dist_01_prc_2 is starting this frame...
            if dist_01_prc_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_01_prc_2.frameNStart = frameN  # exact frame index
                dist_01_prc_2.tStart = t  # local t and not account for scr refresh
                dist_01_prc_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_01_prc_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_01_prc_2.started')
                # update status
                dist_01_prc_2.status = STARTED
                dist_01_prc_2.setAutoDraw(True)
            
            # if dist_01_prc_2 is active this frame...
            if dist_01_prc_2.status == STARTED:
                # update params
                pass
            
            # *dist_02_prc_2* updates
            
            # if dist_02_prc_2 is starting this frame...
            if dist_02_prc_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_02_prc_2.frameNStart = frameN  # exact frame index
                dist_02_prc_2.tStart = t  # local t and not account for scr refresh
                dist_02_prc_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_02_prc_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_02_prc_2.started')
                # update status
                dist_02_prc_2.status = STARTED
                dist_02_prc_2.setAutoDraw(True)
            
            # if dist_02_prc_2 is active this frame...
            if dist_02_prc_2.status == STARTED:
                # update params
                pass
            
            # *correct_prc_2* updates
            
            # if correct_prc_2 is starting this frame...
            if correct_prc_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                correct_prc_2.frameNStart = frameN  # exact frame index
                correct_prc_2.tStart = t  # local t and not account for scr refresh
                correct_prc_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correct_prc_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'correct_prc_2.started')
                # update status
                correct_prc_2.status = STARTED
                correct_prc_2.setAutoDraw(True)
            
            # if correct_prc_2 is active this frame...
            if correct_prc_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback_learn.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_learn.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_learn" ---
        for thisComponent in feedback_learn.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_learn
        feedback_learn.tStop = globalClock.getTime(format='float')
        feedback_learn.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_learn.stopped', feedback_learn.tStop)
        # the Routine "feedback_learn" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if practice_trials.trialList in ([], [None], None):
        params = []
    else:
        params = practice_trials.trialList[0].keys()
    # save data for this loop
    practice_trials.saveAsExcel(filename + '.xlsx', sheetName='practice_trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "instructions_start" ---
    # create an object to store info about Routine instructions_start
    instructions_start = data.Routine(
        name='instructions_start',
        components=[instruction_part4, continue_button_5],
    )
    instructions_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from instruction_part4_text
    if language == "english": 
        instruction_part4.text = (
        "This is the end of the practice trials. Now the "
        "experiment starts. Remember to choose the next image in the "
        "sequence out of the three options. Try to remember the correct orders. "
        "Press any button to continue."
        )
    
        
        
    if language == "german": 
        instruction_part4.text = (
        "Das ist das Ende der Übungsdurchgänge. Jetzt beginnt "
        "das Experiment. Denken Sie daran, das nächste Bild in der "
        "Abfolge aus den drei Optionen auszuwählen. Versuchen Sie, sich die richtige Reihenfolge zu merken. "
        "Drücken Sie eine beliebige Taste, um fortzufahren."
    )
    
    
    # create starting attributes for continue_button_5
    continue_button_5.keys = []
    continue_button_5.rt = []
    _continue_button_5_allKeys = []
    # store start times for instructions_start
    instructions_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_start.tStart = globalClock.getTime(format='float')
    instructions_start.status = STARTED
    thisExp.addData('instructions_start.started', instructions_start.tStart)
    instructions_start.maxDuration = None
    # keep track of which components have finished
    instructions_startComponents = instructions_start.components
    for thisComponent in instructions_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_start" ---
    instructions_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from bids_instruc_4
        bids.schedule_onset(instruction_part4,
                                type_of_stimulus="instructions",
                                component_label="instruction_part4")
        
        # *instruction_part4* updates
        
        # if instruction_part4 is starting this frame...
        if instruction_part4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_part4.frameNStart = frameN  # exact frame index
            instruction_part4.tStart = t  # local t and not account for scr refresh
            instruction_part4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_part4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_part4.started')
            # update status
            instruction_part4.status = STARTED
            instruction_part4.setAutoDraw(True)
        
        # if instruction_part4 is active this frame...
        if instruction_part4.status == STARTED:
            # update params
            pass
        
        # *continue_button_5* updates
        waitOnFlip = False
        
        # if continue_button_5 is starting this frame...
        if continue_button_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_button_5.frameNStart = frameN  # exact frame index
            continue_button_5.tStart = t  # local t and not account for scr refresh
            continue_button_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_button_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_button_5.started')
            # update status
            continue_button_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_button_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_button_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_button_5.status == STARTED and not waitOnFlip:
            theseKeys = continue_button_5.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
            _continue_button_5_allKeys.extend(theseKeys)
            if len(_continue_button_5_allKeys):
                continue_button_5.keys = _continue_button_5_allKeys[-1].name  # just the last key pressed
                continue_button_5.rt = _continue_button_5_allKeys[-1].rt
                continue_button_5.duration = _continue_button_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_start" ---
    for thisComponent in instructions_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_start
    instructions_start.tStop = globalClock.getTime(format='float')
    instructions_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_start.stopped', instructions_start.tStop)
    # Run 'End Routine' code from bids_instruc_4
    bids.mark_offset(instruction_part4) 
    # check responses
    if continue_button_5.keys in ['', [], None]:  # No response was made
        continue_button_5.keys = None
    thisExp.addData('continue_button_5.keys',continue_button_5.keys)
    if continue_button_5.keys != None:  # we had a response
        thisExp.addData('continue_button_5.rt', continue_button_5.rt)
        thisExp.addData('continue_button_5.duration', continue_button_5.duration)
    thisExp.nextEntry()
    # the Routine "instructions_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=3.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('sequences/main_conditions_piloteditionxlsx.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "set_tracking_parameters" ---
        # create an object to store info about Routine set_tracking_parameters
        set_tracking_parameters = data.Routine(
            name='set_tracking_parameters',
            components=[],
        )
        set_tracking_parameters.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from set_parameters
        
        # important parameters to store in the.tsv file for every
        # trial of the learning task
        sequence_name = learningSeq
        trial_num = currPosInSeq
        
        # increment the route counter ONLY at the first trial of each 6-trial route
        if trial_num == 1:
            if sequence_name == "A":
                A_route += 1
            elif sequence_name == "B":
                B_route += 1
            elif sequence_name == "C":
                C_route += 1
        
        # set route_num for this trial based on the current sequence
        if sequence_name == "A":
            route_num = A_route
        elif sequence_name == "B":
            route_num = B_route
        else:  # "C"
            route_num = C_route
        
        # decide the block based on your design:
        # Block 1: A routes 1–10
        # Block 2: A routes 11–20, B routes 1–10
        # Block 3: A routes 21–30, B routes 11–20, C routes 1–10
        if sequence_name == "A":
            if route_num <= 10:
                block_name = "Block 1"
            elif route_num <= 20:
                block_name = "Block 2"
            else:
                block_name = "Block 3"
        elif sequence_name == "B":
            if route_num <= 10:
                block_name = "Block 2"
            else:
                block_name = "Block 3"
        else:  # "C"
            block_name = "Block 3"
        # Run 'Begin Routine' code from log_trial_numbers
        
        # log("current block", block_name)
        log("current sequence: ", sequence_name)
        log("current route: ", route_num)
        log("trial number: ", trial_num)
        # store start times for set_tracking_parameters
        set_tracking_parameters.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        set_tracking_parameters.tStart = globalClock.getTime(format='float')
        set_tracking_parameters.status = STARTED
        thisExp.addData('set_tracking_parameters.started', set_tracking_parameters.tStart)
        set_tracking_parameters.maxDuration = None
        # keep track of which components have finished
        set_tracking_parametersComponents = set_tracking_parameters.components
        for thisComponent in set_tracking_parameters.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "set_tracking_parameters" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        set_tracking_parameters.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                set_tracking_parameters.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in set_tracking_parameters.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "set_tracking_parameters" ---
        for thisComponent in set_tracking_parameters.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for set_tracking_parameters
        set_tracking_parameters.tStop = globalClock.getTime(format='float')
        set_tracking_parameters.tStopRefresh = tThisFlipGlobal
        thisExp.addData('set_tracking_parameters.stopped', set_tracking_parameters.tStop)
        # the Routine "set_tracking_parameters" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "choice_display" ---
        # create an object to store info about Routine choice_display
        choice_display = data.Routine(
            name='choice_display',
            components=[prompt, dist_01, dist_02, correct, key_resp, instructions_choose],
        )
        choice_display.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from bids_choice_disp
        
        # which item sits where on THIS trial
        pos2item = {}
        pos2item[correct_pos]   = 'target'
        pos2item[dist01_pos] = 'distractor_01'
        pos2item[dist02_pos] = 'distractor_02'
        
        # key -> position mapping
        key2pos = {
            left_key:   'left',
            right_key:  'right',
            center_key: 'center'
        }
        prompt.setPos(prompt_pos)
        prompt.setImage('stimuli\\' + promptFile)
        dist_01.setPos([resolve_pos(dist01_pos)])
        dist_01.setImage('stimuli\\' + dist_01File)
        dist_02.setPos([resolve_pos(dist02_pos)])
        dist_02.setImage('stimuli\\' + dist_02File)
        correct.setPos([resolve_pos(correct_pos)])
        correct.setImage('stimuli\\' + correctFile)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from get_response_parameters
        
        
        # determine which one is correct response on this trial 
        # might need later 
        correctAnsKey = pos_to_key[resolve_pos(correct_pos)]
        
        # start a clock to time out routine if necessary
        routClock = core.Clock()
        routClock.reset()
        
        # initialize
        responded = False
        key_resp.keys = []
        key_resp.rt = None
        
        # this attribute needs to also exist otherwise there is error
        key_resp.duration = None
        
        # Run 'Begin Routine' code from instructions_choose_text
        if language == "english":
            instructions_choose.text = (
            "Choose the next image in the sequence!")
         
        if language == "german":
            instructions_choose.text = (
            "Wähle das nächste Bild in der Sequenz!")
        # store start times for choice_display
        choice_display.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        choice_display.tStart = globalClock.getTime(format='float')
        choice_display.status = STARTED
        thisExp.addData('choice_display.started', choice_display.tStart)
        choice_display.maxDuration = None
        # keep track of which components have finished
        choice_displayComponents = choice_display.components
        for thisComponent in choice_display.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "choice_display" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        choice_display.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from bids_choice_disp
            ## schedule bids trigger setting
            bids.schedule_onset(prompt, type_of_stimulus="image", 
            component_label="current_image", 
            concept_label = promptFile.partition("//")[0],
            concept_examplar = promptFile.partition("//")[2], 
            block_name=block_name,
            sequence_name=sequence_name,
            route_num=route_num, 
            trial_num=trial_num)
            
            bids.schedule_onset(correct, type_of_stimulus="image", 
            component_label="corr_next_image", 
            concept_label = correctFile.partition("//")[0],
            concept_examplar = correctFile.partition("//")[2], 
            block_name=block_name,
            sequence_name=sequence_name,
            route_num=route_num, 
            trial_num=trial_num)
            
            bids.schedule_onset(dist_01, type_of_stimulus="image", 
            component_label="distractor 01 (same seq)", 
            concept_label = dist_01File.partition("//")[0],
            concept_examplar = dist_01File.partition("//")[2], 
            block_name=block_name,
            sequence_name=sequence_name,
            route_num=route_num, 
            trial_num=trial_num)
            
            bids.schedule_onset(dist_02, type_of_stimulus="image", 
            component_label="distractor 02 (diff seq)", 
            concept_label = dist_02File.partition("//")[0],
            concept_examplar = dist_02File.partition("//")[2], 
            block_name=block_name,
            sequence_name=sequence_name,
            route_num=route_num, 
            trial_num=trial_num)
            
            # Run 'Each Frame' code from trigger_choice_disp
            # prompt triggering 
            if prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                send_onset_trigger(prompt, promptTrig)
             
            # dist01 triggering 
            if dist_01.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                send_onset_trigger(dist_01, dist_01Trig)
            
            # dist02 triggering 
            if dist_02.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                send_onset_trigger(dist_02, dist_02Trig)
            
            # correct triggering 
            if correct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                send_onset_trigger(correct, correctTrig)
            
            # "choose prompt" triggering 
            if instructions_choose.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                send_onset_trigger(instructions_choose, 32)
            
            # *prompt* updates
            
            # if prompt is starting this frame...
            if prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompt.frameNStart = frameN  # exact frame index
                prompt.tStart = t  # local t and not account for scr refresh
                prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt.started')
                # update status
                prompt.status = STARTED
                prompt.setAutoDraw(True)
            
            # if prompt is active this frame...
            if prompt.status == STARTED:
                # update params
                pass
            
            # *dist_01* updates
            
            # if dist_01 is starting this frame...
            if dist_01.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_01.frameNStart = frameN  # exact frame index
                dist_01.tStart = t  # local t and not account for scr refresh
                dist_01.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_01, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_01.started')
                # update status
                dist_01.status = STARTED
                dist_01.setAutoDraw(True)
            
            # if dist_01 is active this frame...
            if dist_01.status == STARTED:
                # update params
                pass
            
            # *dist_02* updates
            
            # if dist_02 is starting this frame...
            if dist_02.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_02.frameNStart = frameN  # exact frame index
                dist_02.tStart = t  # local t and not account for scr refresh
                dist_02.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_02, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_02.started')
                # update status
                dist_02.status = STARTED
                dist_02.setAutoDraw(True)
            
            # if dist_02 is active this frame...
            if dist_02.status == STARTED:
                # update params
                pass
            
            # *correct* updates
            
            # if correct is starting this frame...
            if correct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                correct.frameNStart = frameN  # exact frame index
                correct.tStart = t  # local t and not account for scr refresh
                correct.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correct, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'correct.started')
                # update status
                correct.status = STARTED
                correct.setAutoDraw(True)
            
            # if correct is active this frame...
            if correct.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('key_resp.started', t)
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                key_resp.clock.reset()  # now t=0
                key_resp.clearEvents(eventType='keyboard')
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + max_response_time-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('key_resp.stopped', t)
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED:
                theseKeys = key_resp.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct_ans)) or (key_resp.keys == correct_ans):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from get_response_parameters
            # timeout case
            if routClock.getTime() >= max_response_time:
                # no key was pressed in time
                key_resp.keys = None
                key_resp.rt = None
                responded = False
                continueRoutine = False 
                key_resp.corr = None
                # (optionally store that it was a timeout)
                log('response too slow')
                
            # if there is key selection 
            if key_resp.status == STARTED:
                theseKeys = key_resp.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
                print(len(theseKeys))
                if len(theseKeys):
                    key_resp.keys     = theseKeys[0].name
                    key_resp.rt       = theseKeys[0].rt
                    key_resp.duration = theseKeys[0].duration  # safe to save later
                    key_resp.corr = 1 if key_resp.keys == correctAnsKey else 0
                    responded = True
                    continueRoutine = False
                    log('response, correct: ', key_resp.corr)
            
             
            
            # *instructions_choose* updates
            
            # if instructions_choose is starting this frame...
            if instructions_choose.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instructions_choose.frameNStart = frameN  # exact frame index
                instructions_choose.tStart = t  # local t and not account for scr refresh
                instructions_choose.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instructions_choose, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions_choose.started')
                # update status
                instructions_choose.status = STARTED
                instructions_choose.setAutoDraw(True)
            
            # if instructions_choose is active this frame...
            if instructions_choose.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                choice_display.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in choice_display.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "choice_display" ---
        for thisComponent in choice_display.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for choice_display
        choice_display.tStop = globalClock.getTime(format='float')
        choice_display.tStopRefresh = tThisFlipGlobal
        thisExp.addData('choice_display.stopped', choice_display.tStop)
        # Run 'End Routine' code from bids_choice_disp
        
        bids.mark_offset(prompt)
        bids.mark_offset(correct)
        bids.mark_offset(dist_01)
        bids.mark_offset(dist_02)
        
        # response neeeds to be saved at end of trial
        pressed = key_resp.keys
        if isinstance(pressed, list):
            pressed = pressed[-1] if pressed else None
        
        chosen_pos  = key2pos.get(pressed, None)
        chosen_item = pos2item.get(chosen_pos, None)
        
        is_correct = (chosen_item == 'target') if chosen_item is not None else None
        
        bids.add_instant(
            "choice",
            response=(key_resp.keys if key_resp.keys else None),
            response_time=(key_resp.rt if hasattr(key_resp, "rt") else None),
            correct=is_correct,
            expected_response = correct_ans, 
            response_meaning = chosen_item
        )
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct_ans).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp.keys',key_resp.keys)
        trials.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # the Routine "choice_display" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[prompt_2, dist_01_2, dist_02_2, correct_2],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from bids_feedback_disp
        
        
        # Run 'Begin Routine' code from animation_control
        
        # Initialize animation control
        endY = prompt_pos[1]    # end y-coor of moving image
        endX = prompt_pos[0]    # end x-coord of moving image
        
        maxAnimationDur = int(round(animation_time * expInfo['frameRate']))
        animationTimer = 0      # initialize variable 
        animationDone = False   # initialize variable 
        moveCorrect = False     # initialize variable 
        # Run 'Begin Routine' code from trigger_feedback_disp
        feedback.trigger_sent = False
        prompt_2.setPos(prompt_pos)
        prompt_2.setImage('stimuli\\'  + promptFile)
        dist_01_2.setPos([resolve_pos(dist01_pos)])
        dist_01_2.setImage('stimuli\\' + dist_01File)
        dist_02_2.setPos([resolve_pos(dist02_pos)])
        dist_02_2.setImage('stimuli\\'  + dist_02File)
        correct_2.setPos([resolve_pos(correct_pos)])
        correct_2.setImage('stimuli\\'  + correctFile)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from bids_feedback_disp
            ## schedule bids trigger setting
            bids.schedule_onset(correct_2, type_of_stimulus="feedback", 
            component_label="correct_moving")
            
            # Run 'Each Frame' code from animation_control
            # Start animation 
            if not moveCorrect and not animationDone:
                moveCorrect = True 
            
            # Run animation
            if moveCorrect and not animationDone:
                animationTimer += 1
                
                # Current position
                current_x, current_y = correct_2.pos
                
                # Compute direction toward target
                dx = endX - current_x
                dy = endY - current_y
            
                # Move a small fraction toward target
                move_fraction = feedback_steps # % of the remaining distance each frame
                new_x = current_x + dx * move_fraction
                new_y = current_y + dy * move_fraction
            
                # Update position
                correct_2.setPos((new_x, new_y))
            
                # Stop when close enough to target
                if (abs(dx) < rest_jump and abs(dy) < rest_jump) or animationTimer > maxAnimationDur:
                # if (abs(dx) < rest_jump and abs(dy) < rest_jump):
                    correct_2.setPos((endX, endY))
                    animationDone = True
                    moveCorrect = False
                    continueRoutine = False
            # Run 'Each Frame' code from trigger_feedback_disp
            # feedback onset
            if moveCorrect and not feedback.trigger_sent: 
                send_trigger(98)
                feedback.trigger_sent = True 
             
            # feedback offset
            if animationDone: 
                send_trigger(99)
            
            
            # *prompt_2* updates
            
            # if prompt_2 is starting this frame...
            if prompt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompt_2.frameNStart = frameN  # exact frame index
                prompt_2.tStart = t  # local t and not account for scr refresh
                prompt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_2.started')
                # update status
                prompt_2.status = STARTED
                prompt_2.setAutoDraw(True)
            
            # if prompt_2 is active this frame...
            if prompt_2.status == STARTED:
                # update params
                pass
            
            # *dist_01_2* updates
            
            # if dist_01_2 is starting this frame...
            if dist_01_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_01_2.frameNStart = frameN  # exact frame index
                dist_01_2.tStart = t  # local t and not account for scr refresh
                dist_01_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_01_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_01_2.started')
                # update status
                dist_01_2.status = STARTED
                dist_01_2.setAutoDraw(True)
            
            # if dist_01_2 is active this frame...
            if dist_01_2.status == STARTED:
                # update params
                pass
            
            # *dist_02_2* updates
            
            # if dist_02_2 is starting this frame...
            if dist_02_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dist_02_2.frameNStart = frameN  # exact frame index
                dist_02_2.tStart = t  # local t and not account for scr refresh
                dist_02_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dist_02_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dist_02_2.started')
                # update status
                dist_02_2.status = STARTED
                dist_02_2.setAutoDraw(True)
            
            # if dist_02_2 is active this frame...
            if dist_02_2.status == STARTED:
                # update params
                pass
            
            # *correct_2* updates
            
            # if correct_2 is starting this frame...
            if correct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                correct_2.frameNStart = frameN  # exact frame index
                correct_2.tStart = t  # local t and not account for scr refresh
                correct_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correct_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'correct_2.started')
                # update status
                correct_2.status = STARTED
                correct_2.setAutoDraw(True)
            
            # if correct_2 is active this frame...
            if correct_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # Run 'End Routine' code from bids_feedback_disp
        #bids.add_instant(
        #    "feedback",
        #    feedback_duration = animationTimer
        #)
        
        bids.mark_offset(correct_2)
        # the Routine "feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "task_break" ---
        # create an object to store info about Routine task_break
        task_break = data.Routine(
            name='task_break',
            components=[breaks_instruction],
        )
        task_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from definition_breaks
        # only break at the break points, otherwise don't continue this routine
        continueRoutine = (trials.thisN in break_points)
        # store start times for task_break
        task_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task_break.tStart = globalClock.getTime(format='float')
        task_break.status = STARTED
        thisExp.addData('task_break.started', task_break.tStart)
        task_break.maxDuration = None
        # keep track of which components have finished
        task_breakComponents = task_break.components
        for thisComponent in task_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task_break" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        task_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *breaks_instruction* updates
            
            # if breaks_instruction is starting this frame...
            if breaks_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breaks_instruction.frameNStart = frameN  # exact frame index
                breaks_instruction.tStart = t  # local t and not account for scr refresh
                breaks_instruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breaks_instruction, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breaks_instruction.started')
                # update status
                breaks_instruction.status = STARTED
                breaks_instruction.setAutoDraw(True)
            
            # if breaks_instruction is active this frame...
            if breaks_instruction.status == STARTED:
                # update params
                pass
            
            # if breaks_instruction is stopping this frame...
            if breaks_instruction.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > breaks_instruction.tStartRefresh + break_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    breaks_instruction.tStop = t  # not accounting for scr refresh
                    breaks_instruction.tStopRefresh = tThisFlipGlobal  # on global time
                    breaks_instruction.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'breaks_instruction.stopped')
                    # update status
                    breaks_instruction.status = FINISHED
                    breaks_instruction.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task_break" ---
        for thisComponent in task_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task_break
        task_break.tStop = globalClock.getTime(format='float')
        task_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task_break.stopped', task_break.tStop)
        # the Routine "task_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "instructions_end" ---
    # create an object to store info about Routine instructions_end
    instructions_end = data.Routine(
        name='instructions_end',
        components=[instruction_end, continue_button_3],
    )
    instructions_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from instructions_end_text
    if language == "english": 
        instruction_part3.text = (
            "This was the experiment. Thank you for your participation! "
            "Press any button to end the task. "
        )
    
        
        
    if language == "german": 
        instruction_part3.text = (
            "Hiermit ist das Experiment beendet. Vielen Dank für Ihre Teilnahme. "
            "Drücken Sie irgendeinen Knopf, um das Experiment zu beenden. "
        )
    
    
    # create starting attributes for continue_button_3
    continue_button_3.keys = []
    continue_button_3.rt = []
    _continue_button_3_allKeys = []
    # store start times for instructions_end
    instructions_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_end.tStart = globalClock.getTime(format='float')
    instructions_end.status = STARTED
    thisExp.addData('instructions_end.started', instructions_end.tStart)
    instructions_end.maxDuration = None
    # keep track of which components have finished
    instructions_endComponents = instructions_end.components
    for thisComponent in instructions_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_end" ---
    instructions_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from bids_instruc_5
        bids.schedule_onset(instruction_end,
                                type_of_stimulus="instructions",
                                component_label="instruction_end")
        
        # *instruction_end* updates
        
        # if instruction_end is starting this frame...
        if instruction_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_end.frameNStart = frameN  # exact frame index
            instruction_end.tStart = t  # local t and not account for scr refresh
            instruction_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_end.started')
            # update status
            instruction_end.status = STARTED
            instruction_end.setAutoDraw(True)
        
        # if instruction_end is active this frame...
        if instruction_end.status == STARTED:
            # update params
            pass
        
        # *continue_button_3* updates
        waitOnFlip = False
        
        # if continue_button_3 is starting this frame...
        if continue_button_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_button_3.frameNStart = frameN  # exact frame index
            continue_button_3.tStart = t  # local t and not account for scr refresh
            continue_button_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_button_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_button_3.started')
            # update status
            continue_button_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_button_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_button_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_button_3.status == STARTED and not waitOnFlip:
            theseKeys = continue_button_3.getKeys(keyList=[left_key, center_key, right_key], ignoreKeys=["escape"], waitRelease=True)
            _continue_button_3_allKeys.extend(theseKeys)
            if len(_continue_button_3_allKeys):
                continue_button_3.keys = _continue_button_3_allKeys[-1].name  # just the last key pressed
                continue_button_3.rt = _continue_button_3_allKeys[-1].rt
                continue_button_3.duration = _continue_button_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_end" ---
    for thisComponent in instructions_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_end
    instructions_end.tStop = globalClock.getTime(format='float')
    instructions_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_end.stopped', instructions_end.tStop)
    # Run 'End Routine' code from bids_instruc_5
    bids.mark_offset(instruction_end) 
    # check responses
    if continue_button_3.keys in ['', [], None]:  # No response was made
        continue_button_3.keys = None
    thisExp.addData('continue_button_3.keys',continue_button_3.keys)
    if continue_button_3.keys != None:  # we had a response
        thisExp.addData('continue_button_3.rt', continue_button_3.rt)
        thisExp.addData('continue_button_3.duration', continue_button_3.duration)
    thisExp.nextEntry()
    # the Routine "instructions_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from trigger_end
    # set trigger that exp has ended
    send_trigger(255)
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='comma')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
