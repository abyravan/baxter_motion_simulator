#!/usr/bin/env python

# Based on file joint_recorder.py from "baxter_examples/scripts"

import argparse

import rospy

import baxter_interface
from baxter_motion_simulator import JointStateRecorder

from baxter_interface import CHECK_VERSION


def main():
    """Joint State Recorder

    Record timestamped joint positions, velocities and efforts to separate files for
    later play back.

    Run this example while moving the robot's arms and grippers
    to record a time series of joint states to the folder at the given path
    Folder name will be appended by current date and time.

    Positions will be saved at folderpath/positions.csv
    Velocities will be saved at folderpath/velocities.csv
    Efforts will be saved at folderpath/efforts.csv

    You can later play the movements back using one of the
    *_file_playback examples.
    """
    epilog = """
Related examples:
  joint_position_file_playback.py; joint_trajectory_file_playback.py.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-f', '--foldername', dest='foldername', required=True,
        help='name of the folder to record data at. Will be appended by current date & time (can have a path)'
    )
    parser.add_argument(
        '-r', '--record-rate', type=int, default=100, metavar='RECORDRATE',
        help='rate at which to record (default: 100)'
    )
    parser.add_argument(
        '-a', '--append-dt', type=int, default=1, metavar='APPENDDT',
        help='append current date and time to folder name (default: True)'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("joint_state_recorder")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print("Enabling robot... ")
    rs.enable()

    recorder = JointStateRecorder(args.foldername, args.record_rate, args.append_dt)
    rospy.on_shutdown(recorder.stop)

    print("Recording. Press Ctrl-C to stop.")
    recorder.record()

    print("\nDone.")

if __name__ == '__main__':
    main()

