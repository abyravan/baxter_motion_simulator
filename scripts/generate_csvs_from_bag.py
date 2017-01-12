#!/usr/bin/env python

# General
import argparse
import math
import random
import time
import os

# ROS
import rospy
import rosbag
from std_msgs.msg import (
    UInt16,
)

# Baxter
import baxter_interface
from baxter_interface import CHECK_VERSION

# Msgs
from sensor_msgs.msg import JointState
from lula_controller_msgs.msg import TaskSpaceCommand
from baxter_core_msgs.msg import JointCommand
from trajectory_msgs.msg import JointTrajectoryPoint

class StateRecorder(object):

    ##### Initialize stuff
    def __init__(self, savefoldername='data', append_dt=True, actjtnames=[], comjtnames=[]):

        # Get folder name
        self._savefoldername  = savefoldername
        if append_dt:
            self._savefoldername  = savefoldername + time.strftime("_%Y_%m_%d_%H_%M_%S")

        # Create the folder to save data at
        if not os.path.exists(self._savefoldername):
            print("Creating directory: [%s] to record joint data" %(self._savefoldername));
            os.makedirs(self._savefoldername)
        else:
            print("Directory: [%s] exists! Will begin OVERWRITING recorded joint data!" %(self._savefoldername));

        ## ACTUAL joint data
        # Setup the filename to save timestamped joint angles
        # First row is the name of the joints
        self._actjtnames = actjtnames;
        print('Recording [actual] joint positions at: ' + self._savefoldername + '/positions.csv')
        self._posf = open(self._savefoldername + '/positions.csv', 'w')
        self._posf.write('time,')
        self._posf.write(','.join([j for j in actjtnames]) + '\n')

        # Setup the filename to save timestamped joint velocities
        # First row is the name of the joints
        print('Recording [actual] joint velocities at: ' + self._savefoldername + '/velocities.csv')
        self._velf = open(self._savefoldername + '/velocities.csv', 'w')
        self._velf.write('time,')
        self._velf.write(','.join([j for j in actjtnames]) + '\n')

        # Setup the filename to save timestamped joint efforts
        # First row is the name of the joints
        print('Recording [actual] joint efforts at: ' + self._savefoldername + '/efforts.csv')
        self._eftf = open(self._savefoldername + '/efforts.csv', 'w')
        self._eftf.write('time,')
        self._eftf.write(','.join([j for j in actjtnames]) + '\n')

        ## COMMANDED joint data
        # Setup the filename to save timestamped joint positions
        # First row is the name of the joints
        self._comjtnames = comjtnames;
        print('Recording [commanded] joint positions at: ' + self._savefoldername + '/commandedpositions.csv')
        self._composf = open(self._savefoldername + '/commandedpositions.csv', 'w')
        self._composf.write('time,')
        self._composf.write(','.join([j for j in comjtnames]) + '\n')

        # Setup the filename to save timestamped joint velocities
        # First row is the name of the joints
        print('Recording [commanded] joint velocities at: ' + self._savefoldername + '/commandedvelocities.csv')
        self._comvelf = open(self._savefoldername + '/commandedvelocities.csv', 'w')
        self._comvelf.write('time,')
        self._comvelf.write(','.join([j for j in comjtnames]) + '\n')

        # Setup the filename to save timestamped joint accelerations
        # First row is the name of the joints
        print('Recording [commanded] joint accelerations at: ' + self._savefoldername + '/commandedaccelerations.csv')
        self._comaccf = open(self._savefoldername + '/commandedaccelerations.csv', 'w')
        self._comaccf.write('time,')
        self._comaccf.write(','.join([j for j in comjtnames]) + '\n')

        ## COMMANDED end effector 3D positions and velocities
        # Setup the filename to save end effector positions
        print('Recording [commanded] end effector positions at: ' + self._savefoldername + '/commandedendeffpositions.csv')
        self._comendefff = open(self._savefoldername + '/commandedendeffpositions.csv', 'w')
        self._comendefff.write('time,x,y,z\n')

    ##### Record an actual sample
    def record_actual_sample(self, msg, t):
        # Check size
        if (len(msg.position) != len(self._actjtnames)):
            return;

        # Get joint angles and write to file
        self._posf.write("%f," % (t.to_sec()))
        self._posf.write(','.join([str(x) for x in msg.position]) + '\n')

        # Get joint velocities and write to file (gripper has no velocities)
        self._velf.write("%f," % (t.to_sec()))
        self._velf.write(','.join([str(x) for x in msg.velocity]) + '\n')

        # Get joint efforts and write to file (gripper writes the force on it)
        self._eftf.write("%f," % (t.to_sec()))
        self._eftf.write(','.join([str(x) for x in msg.effort]) + '\n')

    ##### Close recorder files before exiting
    def finish(self):
        print("Closing recorder files...")
        self._posf.close()
        self._velf.close()
        self._eftf.close()
        self._composf.close()
        self._comvelf.close()
        self._comaccf.close()
        self._comendefff.close()

def main():
    """
    Generates CSV files from bag files with Nathan's data
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-b', '--bagfile', default='', metavar='BAGFILE',
        help='Bag file with the recorded data'
    )
    parser.add_argument(
        '-f', '--savefoldername', default='', metavar='SAVEFOLDERNAME',
        help='Name of the folder to save recorded data at. May be appended by current date & time (can have a path)'
    )
    parser.add_argument(
        '-d', '--append-dt', type=int, default=1, metavar='APPENDDT',
        help='Append current date and time to folder name (default: True)'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    # Open bag file
    bag = rosbag.Bag(args.bagfile);

    # Setup topics we need
    msg_topics = ['/robot/joint_states', '/robot/limb/right/inverse_dynamics_command', '/robot/limb/right/joint_command', '/robot/right_gripper/orig/command'];

    # Get names of joints (actual/commanded)
    actjtnames = []; comjtnames = [];
    for topic, msg, t in bag.read_messages(topics=msg_topics):
        if topic == '/robot/joint_states':
            actjtnames = msg.name;
        elif topic == '/robot/limb/right/joint_command':
            comjtnames = msg.names;
        if (len(actjtnames) != 0) and (len(comjtnames) != 0):
            break;

    # Initialize state recorder
    print("Initializing state recorder")
    gen = StateRecorder(args.savefoldername, args.append_dt, actjtnames, comjtnames);

    # Read bag file
    print("Recording messages")
    ct = 0; maxct = bag.get_message_count();
    for topic, msg, t in bag.read_messages(topics=msg_topics):
        if topic == '/robot/joint_states':
            gen.record_actual_sample(msg, t);
        elif topic == '/robot/limb/right/inverse_dynamics_command':
            # Write commanded velocities
            gen._comvelf.write("%f," % (t.to_sec()))
            gen._comvelf.write(','.join([str(x) for x in msg.velocities]) + '\n')

            # Write commanded accelerations
            gen._comaccf.write("%f," % (t.to_sec()))
            gen._comaccf.write(','.join([str(x) for x in msg.accelerations]) + '\n')
        elif topic == '/robot/limb/right/joint_command':
            # Write commanded positions
            gen._composf.write("%f," % (t.to_sec()))
            gen._composf.write(','.join([str(x) for x in msg.command]) + '\n')
        elif topic == '/robot/right_gripper/orig/command':
            # Write commanded end effector data
            gen._comendefff.write("%f," % (t.to_sec()))
            gen._comendefff.write(','.join([str(x) for x in msg.x]) + '\n')
        ct = ct+1;
        if ((ct % 10000) == 0):
            print('Recording message: [%d/%d]' %(ct,maxct));

    # Finish
    gen.finish();
    print("Finished recording. Number of states: %d"%ct);

if __name__ == '__main__':
    main()
