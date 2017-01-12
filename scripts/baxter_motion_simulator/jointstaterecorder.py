# Joint state recorder - based on recorder.py from "baxter_examples/src/baxter_examples"

import rospy

import baxter_interface

from baxter_interface import CHECK_VERSION

import time
import os

class JointStateRecorder(object):
    def __init__(self, foldername, rate, append_dt=True):
        """
        Records joint state (pos,vel,efforts) to files at the specified folder
        The foldername is appended by current date & time before saving
        """
        # Get folder name
        self._foldername  = foldername
        if append_dt:
            self._foldername  = foldername + time.strftime("_%Y_%m_%d_%H_%M_%S")

        # Setup other vars
        self._posfilename = self._foldername + '/positions.csv';
        self._velfilename = self._foldername + '/velocities.csv';
        self._effortfilename = self._foldername + '/efforts.csv';
        self._raw_rate = rate
        self._rate = rospy.Rate(rate)
        self._start_time = rospy.get_time()
        self._done = False

        # Get the arms
        self._limb_left = baxter_interface.Limb("left")
        self._limb_right = baxter_interface.Limb("right")

        # Get the gripper
        self._gripper_left = baxter_interface.Gripper("left", CHECK_VERSION)
        self._gripper_right = baxter_interface.Gripper("right", CHECK_VERSION)
        self._io_left_lower = baxter_interface.DigitalIO('left_lower_button')
        self._io_left_upper = baxter_interface.DigitalIO('left_upper_button')
        self._io_right_lower = baxter_interface.DigitalIO('right_lower_button')
        self._io_right_upper = baxter_interface.DigitalIO('right_upper_button')

        # Verify Grippers Have No Errors and are Calibrated
        if self._gripper_left.error():
            self._gripper_left.reset()
        if self._gripper_right.error():
            self._gripper_right.reset()
        if (not self._gripper_left.calibrated() and
            self._gripper_left.type() != 'custom'):
            self._gripper_left.calibrate()
        if (not self._gripper_right.calibrated() and
            self._gripper_right.type() != 'custom'):
            self._gripper_right.calibrate()

    def _time_stamp(self):
        return rospy.get_time() - self._start_time

    def stop(self):
        """
        Stop recording.
        """
        self._done = True

    def done(self):
        """
        Return whether or not recording is done.
        """
        if rospy.is_shutdown():
            self.stop()
        return self._done

    def record(self):
        """
        Records the current joint positions, velocities and efforts to their corresponding csv files if outputFilename was
        provided at construction this function will record the latest set of
        joint angles, velocities and efforts in a csv format.

        This function does not test to see if a file exists and will overwrite
        existing files.
        """
        if self._foldername:
            # Create the folder to save the recorded data
            if not os.path.exists(self._foldername):
                print("Creating directory: [%s] to record joint data" %(self._foldername));
                os.makedirs(self._foldername)
            else:
                print("Directory: [%s] exists! Will begin OVERWRITING recorded joint data!" %(self._foldername));

            # Get the names of the joints
            joints_left = self._limb_left.joint_names()
            joints_right = self._limb_right.joint_names()

            # Setup the filename to save timestamped joint angles
            # First row is the name of the joints
            print('Recording joint positions at: ' + self._posfilename)
            p = open(self._posfilename, 'w')
            p.write('time,')
            p.write(','.join([j for j in joints_left]) + ',')
            p.write('left_gripper,')
            p.write(','.join([j for j in joints_right]) + ',')
            p.write('right_gripper\n')

            # Setup the filename to save timestamped joint velocities
            # First row is the name of the joints
            print('Recording joint velocities at: ' + self._posfilename)
            v = open(self._velfilename, 'w')
            v.write('time,')
            v.write(','.join([j for j in joints_left]) + ',')
            v.write('left_gripper,')
            v.write(','.join([j for j in joints_right]) + ',')
            v.write('right_gripper\n')

            # Setup the filename to save timestamped joint velocities
            # First row is the name of the joints
            print('Recording joint efforts at: ' + self._posfilename)
            e = open(self._effortfilename, 'w')
            e.write('time,')
            e.write(','.join([j for j in joints_left]) + ',')
            e.write('left_gripper_force,')
            e.write(','.join([j for j in joints_right]) + ',')
            e.write('right_gripper_force\n')

            # Record till ros shutdown
            while not self.done():
                # Look for gripper button presses
                if self._io_left_lower.state:
                    self._gripper_left.open()
                elif self._io_left_upper.state:
                    self._gripper_left.close()
                if self._io_right_lower.state:
                    self._gripper_right.open()
                elif self._io_right_upper.state:
                    self._gripper_right.close()

                # Get the timestamp
                timestamp   = self._time_stamp()

                # Get joint angles and write to file
                angles_left = [self._limb_left.joint_angle(j)
                               for j in joints_left]
                angles_right = [self._limb_right.joint_angle(j)
                                for j in joints_right]
                p.write("%f," % (timestamp))
                p.write(','.join([str(x) for x in angles_left]) + ',')
                p.write(str(self._gripper_left.position()) + ',')
                p.write(','.join([str(x) for x in angles_right]) + ',')
                p.write(str(self._gripper_right.position()) + '\n')

                # Get joint velocities and write to file (gripper has no velocities)
                velocities_left = [self._limb_left.joint_velocity(j)
                                   for j in joints_left]
                velocities_right = [self._limb_right.joint_velocity(j)
                                    for j in joints_right]
                v.write("%f," % (timestamp))
                v.write(','.join([str(x) for x in velocities_left]) + ',')
                v.write('0,')
                v.write(','.join([str(x) for x in velocities_right]) + ',')
                v.write('0\n')

                # Get joint efforts and write to file (gripper writes the force on it)
                efforts_left = [self._limb_left.joint_effort(j)
                                   for j in joints_left]
                efforts_right = [self._limb_right.joint_effort(j)
                                    for j in joints_right]
                e.write("%f," % (timestamp))
                e.write(','.join([str(x) for x in efforts_left]) + ',')
                e.write(str(self._gripper_left.force()) + ',')
                e.write(','.join([str(x) for x in efforts_right]) + ',')
                e.write(str(self._gripper_right.force()) + '\n')

                self._rate.sleep()

            # Finally close files before exiting
            close(p)
            close(v)
            close(e)
