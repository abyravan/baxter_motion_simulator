#!/usr/bin/env python

# Based on the file "joint_velocity_wobbler.py" from baxter_examples/scripts

# General
import argparse
import math
import random
import time
import os

# ROS
import rospy
from std_msgs.msg import (
    UInt16,
)

# Baxter
import baxter_interface
from baxter_interface import CHECK_VERSION

def clip(value, minval, maxval):
    return max(min(value, maxval), minval)

# Motion generator
class MotionGenerator(object):

    # Initialize stuff
    def __init__(self, arm_to_move='right', savefoldername='data', linear_duration=1.0, nonzeroliststr='1',
                    min_vel=0.5, max_vel=1.5, rate=100, reset_freq=20, append_dt=True):
        """
        Moves one/both arm in a piecewise linear trajectory. Each linear segment lasts for a fixed
        amount of time during which the velocity command is held constant. Can move one, two or many joints at a single time
        """
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)

        # Get the arms
        self._left_arm = baxter_interface.limb.Limb("left")
        self._right_arm = baxter_interface.limb.Limb("right")
        self._left_joint_names = self._left_arm.joint_names()
        self._right_joint_names = self._right_arm.joint_names()

        #### Limits
        # Hard-code min and max angles
        self._left_arm_min_angles  = dict([('left_s0', -1.70168), ('left_s1', -2.147), ('left_e0', -3.05418), ('left_e1', -0.05),
                                           ('left_w0', -3.059), ('left_w1', -1.5708), ('left_w2', -3.059)])
        self._left_arm_max_angles  = dict([('left_s0', 1.70168), ('left_s1', 1.047), ('left_e0', 3.05418), ('left_e1', 2.618),
                                           ('left_w0', 3.059), ('left_w1', 2.094), ('left_w2', 3.059)])
        self._right_arm_min_angles = dict([('right_s0', -1.70168), ('right_s1', -2.147), ('right_e0', -3.05418), ('right_e1', -0.05),
                                           ('right_w0', -3.059), ('right_w1', -1.5708), ('right_w2', -3.059)])
        self._right_arm_max_angles = dict([('right_s0', 1.70168), ('right_s1', 1.047), ('right_e0', 3.05418), ('right_e1', 2.618),
                                           ('right_w0', 3.059), ('right_w1', 2.094), ('right_w2', 3.059)])

        # Shrink limits by 0.5 radians on both min/max vals
        for j in self._left_joint_names:
            self._left_arm_min_angles[j] += 1.0
            self._left_arm_max_angles[j] -= 1.0
        for j in self._right_joint_names:
            self._right_arm_min_angles[j] += 1.0
            self._right_arm_max_angles[j] -= 1.0

        # Passed in params
        self._arm_to_move = arm_to_move # 'right', 'left' or 'both'
        self._linear_duration = linear_duration
        self._min_vel = min_vel
        self._max_vel = max_vel
        self._rate = rate  # Hz
        self._nonzeroliststr = nonzeroliststr
        self._nonzerolist = [int(x) for x in nonzeroliststr.split(',')]
        self._reset_freq = reset_freq # Reset to random pose once every this many motions

        # Initialize velocities to zero
        self._left_commanded_joint_velocities  = dict([(joint, 0.0) for i, joint in enumerate(self._left_joint_names)])
        self._right_commanded_joint_velocities = dict([(joint, 0.0) for i, joint in enumerate(self._right_joint_names)])

        # Initialize robot
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # set joint state publishing to passed in rate (Default: 100 Hz)
        self._pub_rate.publish(self._rate)

        ####### State recorder
        # Get folder name
        self._savefoldername  = savefoldername
        if append_dt:
            self._savefoldername  = savefoldername + time.strftime("_%Y_%m_%d_%H_%M_%S")

        # Setup other vars
        self._start_time = rospy.get_time()

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

    # Reset control modes before exiting
    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._left_arm.exit_control_mode()
            self._right_arm.exit_control_mode()
            self._pub_rate.publish(100)  # 100Hz default joint state rate
            rate.sleep()
        return True

    # Set arms to neutral pose
    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        #self._left_arm.move_to_neutral()
        self._right_arm.move_to_neutral()

    # Generate timestamp for recording data
    def _time_stamp(self):
        return rospy.get_time() - self._start_time

    # Create the folder and csv files to save the recorded data
    def open_recorder_files(self):
        if not os.path.exists(self._savefoldername):
            print("Creating directory: [%s] to record joint data" %(self._savefoldername));
            os.makedirs(self._savefoldername)
        else:
            print("Directory: [%s] exists! Will begin OVERWRITING recorded joint data!" %(self._savefoldername));

        # Setup the filename to save timestamped joint angles
        # First row is the name of the joints
        print('Recording joint positions at: ' + self._savefoldername + '/positions.csv')
        self._posf = open(self._savefoldername + '/positions.csv', 'w')
        self._posf.write('time,')
        self._posf.write(','.join([j for j in self._left_joint_names]) + ',')
        self._posf.write('left_gripper,')
        self._posf.write(','.join([j for j in self._right_joint_names]) + ',')
        self._posf.write('right_gripper\n')

        # Setup the filename to save timestamped joint velocities
        # First row is the name of the joints
        print('Recording joint velocities at: ' + self._savefoldername + '/velocities.csv')
        self._velf = open(self._savefoldername + '/velocities.csv', 'w')
        self._velf.write('time,')
        self._velf.write(','.join([j for j in self._left_joint_names]) + ',')
        self._velf.write('left_gripper,')
        self._velf.write(','.join([j for j in self._right_joint_names]) + ',')
        self._velf.write('right_gripper\n')

        # Setup the filename to save timestamped joint velocities
        # First row is the name of the joints
        print('Recording joint velocities at: ' + self._savefoldername + '/commandedvelocities.csv')
        self._comvelf = open(self._savefoldername + '/commandedvelocities.csv', 'w')
        self._comvelf.write('time,')
        self._comvelf.write(','.join([j for j in self._left_joint_names]) + ',')
        self._comvelf.write('left_gripper,')
        self._comvelf.write(','.join([j for j in self._right_joint_names]) + ',')
        self._comvelf.write('right_gripper\n')

        # Setup the filename to save timestamped joint velocities
        # First row is the name of the joints
        print('Recording joint efforts at: ' + self._savefoldername + '/efforts.csv')
        self._eftf = open(self._savefoldername + '/efforts.csv', 'w')
        self._eftf.write('time,')
        self._eftf.write(','.join([j for j in self._left_joint_names]) + ',')
        self._eftf.write('left_gripper_force,')
        self._eftf.write(','.join([j for j in self._right_joint_names]) + ',')
        self._eftf.write('right_gripper_force\n')

    def write_parameters_to_file(self):
        print('Saving parameters at: ' + self._savefoldername + '/parameters.csv')
        self._parf = open(self._savefoldername + '/parameters', 'w')
        self._parf.write('minvel,maxvel,recordrate,nonzeroliststr,linearduration,resetfreq\n')
        self._parf.write('%f,%f,%f,[%s],%f,%f' % (self._min_vel, self._max_vel,
                self._rate, self._nonzeroliststr, self._linear_duration, self._reset_freq) )
        self._parf.close()

    # Close recorder files before exiting
    def _close_recorder_files(self):
        print("Closing recorder files...")
        self._posf.close()
        self._velf.close()
        self._comvelf.close()
        self._eftf.close()

    # Call before shutdown of node - clean up everything
    def clean_shutdown(self):
        print("\nExiting example...")
        #return to normal
        self._close_recorder_files()
        self._reset_control_modes()
        self.set_neutral()
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

    ########### Different methods for sampling velocity commands
    # Meta function which chooses a certain function to call based on the sampling scheme
    def _sample_velocity(self, dim):
        return self.sample_rand_velocity_scaled(dim, random.choice(self._nonzerolist)) # Choose a random number of dimensions to apply velocity

    def randsign(self):
        return ((random.random() > 0.5) and -1 or 1)

    # Sample a set of velocities
    def sample_rand_velocity(self, dim, nz=1):
        ids = random.sample(range(dim),nz)
        return [((k in ids) and ((self._min_vel + random.uniform(0,self._max_vel-self._min_vel)) * self.randsign()) or 0)
                  for k in xrange(dim)]

    # Sample a set of velocities (scaled based on the joints - joints near end have higher velocity)
    def sample_rand_velocity_scaled(self, dim, nz=1):
        ids = random.sample(range(dim),nz)
        scale = [(((k==5 or k==6) and 1.5) or ((k==3 or k==4) and 1.25) or 1.0) for k in xrange(dim)]
        return [((k in ids) and ((self._min_vel + random.uniform(0,self._max_vel-self._min_vel)) * self.randsign() * scale[k]) or 0)
                for k in xrange(dim)]

    # Sample a new set of velocities for the arms
    def sample_velocities(self):
        # Sample velocity for left arm
        if self._arm_to_move == 'left' or self._arm_to_move == 'both':
            sample = self._sample_velocity(len(self._left_joint_names))
            for i, joint in enumerate(self._left_joint_names):
                self._left_commanded_joint_velocities[joint] = sample[i]

        # Sample velocity for right arm
        if self._arm_to_move == 'right' or self._arm_to_move == 'both':
            sample = self._sample_velocity(len(self._right_joint_names))
            for i, joint in enumerate(self._right_joint_names):
                self._right_commanded_joint_velocities[joint] = sample[i]

    def zero_velocities(self):
        for joint in self._left_commanded_joint_velocities:
            self._left_commanded_joint_velocities[joint] = 0;
        for joint in self._right_commanded_joint_velocities:
            self._right_commanded_joint_velocities[joint] = 0;

    '''
    # Command the current joint angles for a set period of time to zero out any remaining velocities on the arm
    def rest_arms(self):
        rest_start = rospy.Time.now()
        rate = rospy.Rate(self._rate)
        self.zero_velocities()
        left_positions = self._left_arm.joint_angles()
        right_positions = self._right_arm.joint_angles()
        while (rospy.Time.now() - rest_start) < self._resting_duration:
            if self._arm_to_move == 'left' or self._arm_to_move == 'both':
                self._left_arm.set_joint_positions(left_positions)
            if self._arm_to_move == 'right' or self._arm_to_move == 'both':
                self._right_arm.set_joint_positions(right_positions)
            self.record_sample()
            rate.sleep()
    '''

    # Sample a random random pose within the limit volume
    def sample_random_pose(self, min_limits, max_limits):
        return dict([(j, min_limits[j] + random.random() * (max_limits[j] - min_limits[j]))
                     for j in max_limits])

    # Set arms to a random pose in the limit volume defined by min & max joint limits
    def set_arms_to_random_pose(self):
        if self._arm_to_move == 'left' or self._arm_to_move == 'both':
            random_pose = self.sample_random_pose(self._left_arm_min_angles, self._left_arm_max_angles)
            self._left_arm.move_to_joint_positions(random_pose)
        if self._arm_to_move == 'right' or self._arm_to_move == 'both':
            random_pose = self.sample_random_pose(self._right_arm_min_angles, self._right_arm_max_angles)
            self._right_arm.move_to_joint_positions(random_pose)

    def record_sample(self):
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
        angles_left = [self._left_arm.joint_angle(j)
                       for j in self._left_joint_names]
        angles_right = [self._right_arm.joint_angle(j)
                        for j in self._right_joint_names]
        self._posf.write("%f," % (timestamp))
        self._posf.write(','.join([str(x) for x in angles_left]) + ',')
        self._posf.write(str(self._gripper_left.position()) + ',')
        self._posf.write(','.join([str(x) for x in angles_right]) + ',')
        self._posf.write(str(self._gripper_right.position()) + '\n')

        # Get joint velocities and write to file (gripper has no velocities)
        velocities_left = [self._left_arm.joint_velocity(j)
                           for j in self._left_joint_names]
        velocities_right = [self._right_arm.joint_velocity(j)
                            for j in self._right_joint_names]
        self._velf.write("%f," % (timestamp))
        self._velf.write(','.join([str(x) for x in velocities_left]) + ',')
        self._velf.write('0,')
        self._velf.write(','.join([str(x) for x in velocities_right]) + ',')
        self._velf.write('0\n')

        # Get "commanded" joint velocities and write to file (gripper has no velocities)
        commandedvelocities_left = [self._left_commanded_joint_velocities[j]
                            for j in self._left_joint_names]
        commandedvelocities_right = [self._right_commanded_joint_velocities[j]
                            for j in self._right_joint_names]
        self._comvelf.write("%f," % (timestamp))
        self._comvelf.write(','.join([str(x) for x in commandedvelocities_left]) + ',')
        self._comvelf.write('0,')
        self._comvelf.write(','.join([str(x) for x in commandedvelocities_right]) + ',')
        self._comvelf.write('0\n')

        # Get joint efforts and write to file (gripper writes the force on it)
        efforts_left = [self._left_arm.joint_effort(j)
                           for j in self._left_joint_names]
        efforts_right = [self._right_arm.joint_effort(j)
                            for j in self._right_joint_names]
        self._eftf.write("%f," % (timestamp))
        self._eftf.write(','.join([str(x) for x in efforts_left]) + ',')
        self._eftf.write(str(self._gripper_left.force()) + ',')
        self._eftf.write(','.join([str(x) for x in efforts_right]) + ',')
        self._eftf.write(str(self._gripper_right.force()) + '\n')

    # Generate a sequence of motions (forever) for the robot
    def generate_motion(self):
        # Move both arms to neutral position
        self.set_neutral()

        # Create the folder and files for recording data
        self.open_recorder_files()

        # Save the passed in parameters
        self.write_parameters_to_file()

        # Initialize stuff
        rate = rospy.Rate(self._rate)

        # Initialize timer and sample an initial velocities
        num_motions = 1
        start = rospy.Time.now()
        self.sample_velocities()

        # Generate motion
        print("Generating motion. Press Ctrl-C to stop...")
        while not rospy.is_shutdown():
            self._pub_rate.publish(self._rate)

            # Change velocities at the start of each new motion
            elapsed = (rospy.Time.now() - start).to_sec()
            if elapsed > 2*self._linear_duration:
                # Periodically sample a random pose and put the robot there (to avoid being near limits etc)
                num_motions += 1 # Increment counter
                if num_motions % self._reset_freq == 0:
                    self.set_arms_to_random_pose()
                    print('Num motions generated: %d. Reset arms to a random pose.' %(num_motions));

                # Sample new velocities and start a motion again
                start = rospy.Time.now() # Reset timer
                self.sample_velocities()
                print('Motion %d. Sampled velocities: ' %num_motions);
                print(self._left_commanded_joint_velocities)
                print(self._right_commanded_joint_velocities)

            # Three phases - ramp up, constant, ramp_down
            if elapsed >= 1.5 * self._linear_duration:
                # Ramp down = position control to a position in the same direction as the arm motion
                left_positions = self._left_arm.joint_angles()
                right_positions = self._right_arm.joint_angles()
                # Choose a target that is further down in direction of motion (and within joint limits)
                #dt = 0.25 # Move by this much time along velocity direction from current position
                #for j in self._left_joint_names:
                #    left_positions[j] += dt * self._left_arm.joint_velocity(j)
                #for j in self._right_joint_names:
                #    right_positions[j] += dt * self._right_arm.joint_velocity(j)
                # Move to the chosen positions
                if self._arm_to_move == 'left' or self._arm_to_move == 'both':
                    self._left_arm.set_joint_positions(left_positions)
                if self._arm_to_move == 'right' or self._arm_to_move == 'both':
                    self._right_arm.set_joint_positions(right_positions)
                time.sleep(0.5);
            else:
                #Ramp up = slow velocity increase
                if elapsed < 0.5 * self._linear_duration:
                    t = min(elapsed / (0.25 * self._linear_duration), 1.0)
                    left_vel = dict([(j, t * self._left_commanded_joint_velocities[j]) for j in self._left_commanded_joint_velocities])
                    right_vel = dict([(j, t * self._right_commanded_joint_velocities[j]) for j in self._right_commanded_joint_velocities])
                else:
                    # Set constant velocity
                    left_vel  = self._left_commanded_joint_velocities
                    right_vel = self._right_commanded_joint_velocities
                    # Record the data from the robot
                    self.record_sample()

                # Set velocities
                if self._arm_to_move == 'left' or self._arm_to_move == 'both':
                    self._left_arm.set_joint_velocities(left_vel)
                if self._arm_to_move == 'right' or self._arm_to_move == 'both':
                    self._right_arm.set_joint_velocities(right_vel)

            # Sleep for a bit
            rate.sleep()

def main():
    """
    Commands joint velocities of to one or both arms such that they move in
    a piecewise linear trajectory. Each linear part lasts for a fixed amount of time
    during which the velocity is held constant. Can move one, two or many joints at a single time
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-a', '--arm', default='right', metavar='ARM',
         help='arm(s) to move (default: right | left, both)'
    )
    parser.add_argument(
        '-f', '--savefoldername', default='', metavar='SAVEFOLDERNAME',
        help='name of the folder to save recorded data at. Will be appended by current date & time (can have a path)'
    )
    parser.add_argument(
        '-l', '--linear-duration', type=float, default=1.0, metavar='LINEARDURATION',
        help='Duration of the linear motion (default: 1.0 sec)'
    )
    parser.add_argument(
        '-n', '--nonzerolist', default='1', metavar='NONZEROLIST',
        help='Different variants of non-zero dimensions. (Default: "1" means all commands will have 1 joint moving.'
                    '"1,2" = randomly choosing between 1 or 2 joints moving and so on )'
    )
    parser.add_argument(
        '-mi', '--min-vel', type=float, default=0.5, metavar='MINVEL',
        help='Min velocity for any of the commands. (Default: 0.5 rad/sec)'
    )
    parser.add_argument(
        '-ma', '--max-vel', type=float, default=1.5, metavar='MAXVEL',
        help='Max velocity for any of the commands. (Default: 1.5 rad/sec)'
    )
    parser.add_argument(
        '-r', '--record-rate', type=int, default=100, metavar='RECORDRATE',
        help='rate at which to record (default: 100 Hz)'
    )
    parser.add_argument(
        '-rf', '--reset-freq', type=int, default=20, metavar='RESETFREQ',
        help='Reset to random pose once every this many poses (default: 20)'
    )
    parser.add_argument(
        '-d', '--append-dt', type=int, default=1, metavar='APPENDDT',
        help='append current date and time to folder name (default: True)'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("arm_motion_generator")

    gen = MotionGenerator(args.arm, args.savefoldername, args.linear_duration, args.nonzerolist,
                args.min_vel, args.max_vel, args.record_rate, args.reset_freq, args.append_dt)
    rospy.on_shutdown(gen.clean_shutdown)
    gen.generate_motion()

    print("Done.")

if __name__ == '__main__':
    main()


