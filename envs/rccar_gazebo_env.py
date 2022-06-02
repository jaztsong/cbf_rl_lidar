import os
from collections import OrderedDict
import numpy as np
from PIL import Image
from io import BytesIO

from gym import Env
from env_spec import EnvSpec
# from gcg.envs.env_spec import EnvSpec
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
# from gcg.envs.spaces.box import Box
# from gcg.envs.spaces.discrete import Discrete
import logging
# from gcg.data.logger import logger

try:
    import rospy
    import rosbag
    import std_msgs.msg
    import geometry_msgs.msg
    import sensor_msgs.msg
    import ackermann_msgs.msg
    ROS_IMPORTED = True
except:
    ROS_IMPORTED = False


class RolloutRosbag:
    def __init__(self):
        self._rosbag = None
        self._last_write = None

    @property
    def _rosbag_dir(self):
        dir = os.path.join("./", 'rosbags')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    def _rosbag_name(self, num):
        return os.path.join(self._rosbag_dir, 'rosbag{0:04d}.bag'.format(num))

    @property
    def is_open(self):
        return (self._rosbag is not None)

    def open(self):
        assert (not self.is_open)

        bag_num = 0
        while os.path.exists(self._rosbag_name(bag_num)):
            bag_num += 1

        self._rosbag = rosbag.Bag(self._rosbag_name(bag_num), 'w')
        self._last_write = rospy.Time.now()

    def write(self, topic, msg, stamp):
        assert (self._rosbag is not None)

        if msg is not None and stamp is not None:
            if stamp > self._last_write:
                self._rosbag.write(topic, msg)
        else:
            logging.warn('Topic {0} not received'.format(topic))

    def write_all(self, topics, msg_dict, stamp_dict):
        for topic in topics:
            self.write(topic, msg_dict.get(topic), stamp_dict.get(topic))

    def close(self):
        assert (self.is_open)

        self._rosbag.close()
        self._rosbag = None
        self._last_write = None

    def trash(self):
        assert (self.is_open)

        bag_fname = self._rosbag.filename

        try:
            self.close()
        except:
            pass

        os.remove(os.path.join(self._rosbag_dir, bag_fname))


class RccarGazeboEnv(Env):
    def __init__(self, params={}):
        params.setdefault('dt', 0.25)
        params.setdefault('horizon', int(5. * 60. / params['dt']))  # 5 minutes worth
        # params.setdefault('ros_namespace', '/rccar/')
        params.setdefault('ros_namespace', '')
        params.setdefault('obs_shape', (36, 64, 1))
        params.setdefault('steer_limits', [-0.9, 0.9])
        params.setdefault('speed_limits', [0.6, 0.6])
        params.setdefault('backup_motor', -0.4)
        params.setdefault('backup_duration', 3.6)
        params.setdefault('backup_steer_range', (-0.8, 0.8))
        params.setdefault('press_enter_on_reset', False)
        params.setdefault('collision_reward', -1.0)
        params.setdefault('collision_reward_only', True)
        
        self._use_vel = True
        self._obs_shape = params['obs_shape']
        self._steer_limits = params['steer_limits']
        self._speed_limits = params['speed_limits']
        self._fixed_speed = (self._speed_limits[0] == self._speed_limits[1] and self._use_vel)
        self._collision_reward = params['collision_reward']
        self._collision_reward_only = params['collision_reward_only']

        self._dt = params['dt']
        self._horizon = params['horizon']

        self._setup_spec()
        assert (self.observation_im_space.shape[-1] == 1 or self.observation_im_space.shape[-1] == 3)
        self.spec = EnvSpec(
            observation_im_space=self.observation_im_space,
            action_space=self.action_space,
            action_selection_space=self.action_selection_space,
            observation_vec_spec=self.observation_vec_spec,
            action_spec=self.action_spec,
            action_selection_spec=self.action_selection_spec,
            goal_spec=self.goal_spec)

        self._last_step_time = None
        self._is_collision = False
        self._backup_motor = params['backup_motor']
        self._backup_duration = params['backup_duration']
        self._backup_steer_range = params['backup_steer_range']
        self._press_enter_on_reset = params['press_enter_on_reset']

        ### ROS
        if not ROS_IMPORTED:
            logging.error('ROS not imported')
            exit(1)

        self._setup_ros(params)
        self._t = 0

    ##################
    ### Setup spec ###
    ##################

    def _setup_spec(self):
        self._setup_action_spec()
        self._setup_observation_spec()
        self._setup_goal_spec()

    def _setup_action_spec(self):
        self.action_spec = OrderedDict()
        self.action_selection_spec = OrderedDict()

        self.action_spec['steer'] = Box(low=np.array([-1.]), high=np.array([1.]))
        self.action_spec['speed'] = Box(low=np.array([-0.7]), high=np.array([0.7]))
        self.action_space = Box(low=np.array([self.action_spec['steer'].low[0], self.action_spec['speed'].low[0]]),
                                high=np.array([self.action_spec['steer'].high[0], self.action_spec['speed'].high[0]]))

        self.action_selection_spec['steer'] = Box(low=np.array([self._steer_limits[0]]), high=np.array([self._steer_limits[1]]))
        self.action_selection_spec['speed'] = Box(low=np.array([self._speed_limits[0]]), high=np.array([self._speed_limits[1]]))
        self.action_selection_space = Box(low=np.array([self.action_selection_spec['steer'].low[0],
                                                        self.action_selection_spec['speed'].low[0]]),
                                          high=np.array([self.action_selection_spec['steer'].high[0],
                                                         self.action_selection_spec['speed'].high[0]]))

        assert (np.logical_and(self.action_selection_space.low >= self.action_space.low,
                               self.action_selection_space.high <= self.action_space.high).all())

    def _setup_observation_spec(self):
        self.observation_im_space = Box(low=0, high=255, shape=self._obs_shape)

        self.observation_vec_spec = OrderedDict()
        self.observation_vec_spec['coll'] = Discrete(1)
        # self.observation_vec_spec['heading_cos'] = Box(low=-1., high=1.)
        # self.observation_vec_spec['heading_sin'] = Box(low=-1., high=1.)
        self.observation_vec_spec['speed'] = Box(low=np.array([-0.4]), high=np.array([0.4]))
        self.observation_vec_spec['steer'] = Box(low=np.array([-1.]), high=np.array([1.]))
        self.observation_vec_spec['motor'] = Box(low=np.array([-1.]), high=np.array([1.]))

    def _setup_goal_spec(self):
        self.goal_spec = OrderedDict()

    #################
    ### Setup ROS ###
    #################

    def _setup_ros(self, params):
        rospy.init_node('RccarGazeboEnv', anonymous=True)
        rospy.sleep(1)

        self._ros_namespace = params['ros_namespace']
        self._ros_topics_and_types = self._setup_get_ros_topics_and_types()
        self._ros_msgs = dict()
        self._ros_msg_times = dict()
        for topic, type in self._ros_topics_and_types.items():
            rospy.Subscriber(self._ros_namespace + topic, type, self.ros_msg_update, (topic,))
        self._setup_ros_publishers()

        self._ros_rolloutbag = RolloutRosbag()

    def _setup_get_ros_topics_and_types(self):
        # return dict([
        #     ('camera/image_raw/compressed', sensor_msgs.msg.CompressedImage),
        #     ('mode', std_msgs.msg.Int32),
        #     ('steer', std_msgs.msg.Float32),
        #     ('motor', std_msgs.msg.Float32),
        #     ('encoder/left', std_msgs.msg.Float32),
        #     ('encoder/right', std_msgs.msg.Float32),
        #     ('encoder/both', std_msgs.msg.Float32),
        #     ('encoder/error', std_msgs.msg.Int32),
        #     ('orientation/quat', geometry_msgs.msg.Quaternion),
        #     ('orientation/rpy', geometry_msgs.msg.Vector3),
        #     ('imu', geometry_msgs.msg.Accel),
        #     ('collision/all', std_msgs.msg.Int32),
        #     ('collision/flip', std_msgs.msg.Int32),
        #     ('collision/jolt', std_msgs.msg.Int32),
        #     ('collision/stuck', std_msgs.msg.Int32),
        #     ('collision/bumper', std_msgs.msg.Int32),
        #     ('cmd/steer', std_msgs.msg.Float32),
        #     ('cmd/motor', std_msgs.msg.Float32),
        #     ('cmd/vel', std_msgs.msg.Float32)
        # ])

        return dict([
            ('/camera/zed/rgb/image_rect_color/compressed', sensor_msgs.msg.CompressedImage),
            ('/vesc/low_level/ackermann_cmd_mux/output', ackermann_msgs.msg.AckermannDriveStamped),
            ('/vesc/ackermann_cmd_mux/input/teleop', ackermann_msgs.msg.AckermannDriveStamped),
            ('/collision', std_msgs.msg.Int32)
        ])

    def _setup_ros_publishers(self):
        self._ros_drive_pub = rospy.Publisher(self._ros_namespace + '/vesc/ackermann_cmd_mux/input/teleop', ackermann_msgs.msg.AckermannDriveStamped, queue_size=10)

        # self._ros_steer_pub = rospy.Publisher(self._ros_namespace + 'cmd/steer', std_msgs.msg.Float32, queue_size=10)
        # self._ros_vel_pub = rospy.Publisher(self._ros_namespace + 'cmd/vel', std_msgs.msg.Float32, queue_size=10)
        # self._ros_motor_pub = rospy.Publisher(self._ros_namespace + 'cmd/motor', std_msgs.msg.Float32, queue_size=10)
        self._ros_pid_enable_pub = rospy.Publisher(self._ros_namespace + 'pid/enable', std_msgs.msg.Empty,
                                                   queue_size=10)
        self._ros_pid_disable_pub = rospy.Publisher(self._ros_namespace + 'pid/disable', std_msgs.msg.Empty,
                                                    queue_size=10)

    ############
    ### Gets ###
    ############

    def _get_observation(self):
        msg = self._ros_msgs['/camera/zed/rgb/image_rect_color/compressed']

        recon_pil_jpg = BytesIO(msg.data)
        recon_pil_arr = Image.open(recon_pil_jpg)
        im = np.array(recon_pil_arr)  # don't need to resize, replay pool takes care of that

        vec_keys = list(self.observation_vec_spec.keys())
        vec = np.zeros(len(vec_keys), dtype=np.float32)

        vec[vec_keys.index('coll')] = self._is_collision
        # vec[vec_keys.index('heading_cos')] = np.cos(self._ros_msgs['orientation/rpy'].z)
        # vec[vec_keys.index('heading_sin')] = np.sin(self._ros_msgs['orientation/rpy'].z)
        vec[vec_keys.index('speed')] = self._get_speed()
        vec[vec_keys.index('steer')] = self._ros_msgs['/vesc/low_level/ackermann_cmd_mux/output'].drive.steering_angle
        vec[vec_keys.index('motor')] = self._ros_msgs['/vesc/low_level/ackermann_cmd_mux/output'].drive.acceleration

        return im, vec

    def _get_goal(self):
        return np.array([], dtype=np.float32)

    def _get_speed(self):
        return self._ros_msgs['/vesc/low_level/ackermann_cmd_mux/output'].drive.speed

        # return self._ros_msgs['encoder/both'].data

    def _get_reward(self):
        if self._is_collision:
            reward = self._collision_reward
        else:
            if self._collision_reward_only:
                reward = 0
            else:
                reward = self._get_speed()
        return reward

    def _get_done(self):
        return self._is_collision

    ########################
    ### Publish commands ###
    ########################

    def _set_action(self, action):
        # cmd_steer, cmd_vel = action
        # self._set_steer(cmd_steer)
        # self._set_vel(cmd_vel)
        self._set_drive(action)

    def _set_drive(self, action):
        cmd_steer, cmd_vel = action

        msg = ackermann_msgs.msg.AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        msg.drive.speed = cmd_vel
        msg.drive.acceleration = 1
        msg.drive.jerk = 1
        msg.drive.steering_angle = cmd_steer
        msg.drive.steering_angle_velocity = 1

        self._ros_drive_pub.publish(msg)

    # def _set_steer(self, steer):
    #     self._ros_steer_pub.publish(std_msgs.msg.Float32(steer))

    # def _set_vel(self, vel):
    #     if self._is_collision and vel > 0:
    #         vel = 0.
    #     self._ros_vel_pub.publish(std_msgs.msg.Float32(vel))

    # def _set_motor(self, motor):
    #     if self._is_collision and motor > 0:
    #         motor = 0
    #     self._ros_motor_pub.publish(std_msgs.msg.Float32(motor))

    def _do_backup(self):
        if self._backup_duration > 0:
            backup_steer = np.random.uniform(*self._backup_steer_range)

            self._ros_pid_disable_pub.publish(std_msgs.msg.Empty())
            rospy.sleep(0.25)
            start_time = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < self._backup_duration:
                self._set_drive((backup_steer, self._backup_motor))
                rospy.sleep(0.01)

            self._set_drive((backup_steer, 0))
            self._ros_pid_enable_pub.publish(std_msgs.msg.Empty())
            rospy.sleep(0.25)

        self._set_drive((0, 0))

    @property
    def horizon(self):
        return self._horizon

    ############
    ### Step ###
    ############

    def step(self, action, offline=False):
        if not offline:
            assert (self.ros_is_good())

        action = np.asarray(action)
        if not (np.logical_and(action >= self.action_space.low, action <= self.action_space.high).all()):
            logging.warn('Action {0} will be clipped to be within bounds: {1}, {2}'.format(action,
                                                                                          self.action_space.low,
                                                                                          self.action_space.high))
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self._set_action(action)

        if not offline:
            rospy.sleep(max(0., self._dt - (rospy.Time.now() - self._last_step_time).to_sec()))
            self._last_step_time = rospy.Time.now()

        next_observation = self._get_observation()
        goal = self._get_goal()
        reward = self._get_reward()
        done = self._get_done()
        env_info = dict()

        self._t += 1

        if not offline:
            self._ros_rolloutbag.write_all(self._ros_topics_and_types.keys(), self._ros_msgs, self._ros_msg_times)
            if done:
                logging.debug('Done after {0} steps'.format(self._t))
                self._t = 0
                self._ros_rolloutbag.close()

        return next_observation, goal, reward, done, env_info

    #############
    ### Reset ###
    #############

    def reset(self, offline=False, keep_rosbag=False):
        if offline:
            self._is_collision = False
            return self._get_observation(), self._get_goal()

        assert (self.ros_is_good())

        if self._ros_rolloutbag.is_open:
            if keep_rosbag:
                self._ros_rolloutbag.close()
            else:
                # should've been closed in step when done
                logging.debug('Trashing bag')
                self._ros_rolloutbag.trash()

        if self._press_enter_on_reset:
            logging.info('Resetting, press enter to continue')
            input()
        else:
            if self._is_collision:
                logging.debug('Resetting (collision)')
            else:
                logging.debug('Resetting (no collision)')

            self._do_backup()

        rospy.sleep(0.5)

        self._last_step_time = rospy.Time.now()
        self._is_collision = False
        self._t = 0

        self._ros_rolloutbag.open()

        self._reset_goal()

        assert (self.ros_is_good())

        return self._get_observation(), self._get_goal()

    def _reset_goal(self):
        pass

    @property
    def horizon(self):
        return self._horizon
    
    ###########
    ### ROS ###
    ###########

    def ros_msg_update(self, msg, args):
        topic = args[0]

        if 'collision' in topic:
            if msg.data == 1:
                self._is_collision = True

            if self._is_collision:
                if msg.data == 1:
                    # if is_collision and current is collision, update
                    self._ros_msgs[topic] = msg
                    self._ros_msg_times[topic] = rospy.Time.now()
                else:
                    if self._ros_msgs[topic].data != 1:
                        # if is collision, but previous message is not collision, then this topic didn't cause a colision
                        self._ros_msgs[topic] = msg
                        self._ros_msg_times[topic] = rospy.Time.now()
            else:
                # always update if not in collision
                self._ros_msgs[topic] = msg
                self._ros_msg_times[topic] = rospy.Time.now()
        else:
            self._ros_msgs[topic] = msg
            self._ros_msg_times[topic] = rospy.Time.now()

    def ros_is_good(self, print=True):
        # check that all not commands are coming in at a continuous rate
        for topic in self._ros_topics_and_types.keys():
            if 'vesc' not in topic and 'collision' not in topic and 'goal' not in topic:
                if topic not in self._ros_msg_times:
                    if print:
                        logging.debug('Topic {0} has never been received'.format(topic))
                    return False
                elapsed = (rospy.Time.now() - self._ros_msg_times[topic]).to_sec()
                if elapsed > self._dt:
                    if print:
                        logging.debug('Topic {0} was received {1} seconds ago (dt is {2})'.format(topic, elapsed, self._dt))
                    return False

        # check if in python mode
        # if self._ros_msgs.get('mode') is None or self._ros_msgs['mode'].data != 2:
        #     if print:
        #         logging.debug('In mode {0}'.format(self._ros_msgs.get('mode')))
        #     return False

        # TODO
        #if self._ros_msgs['collision/flip'].data:
        #    if print:
        #        logging.warn('Car has flipped, please unflip it to continue')
        #    self._is_collision = False # otherwise will stay flipped forever
        #    return False

        # if self._ros_msgs['encoder/error'].data:
        #     if print:
        #         logging.warn('Encoder error')
        #     return False
        
        return True

    def get_mode(self):
        if 'mode' in self._ros_msgs:
            return self._ros_msgs['mode'].data

    def get_default_goal_dict(self):
        return {}

    def ros_get_action(self, t, ros_bag_dict):
        return np.array([ros_bag_dict['cmd/steer'][t].data, ros_bag_dict['cmd/vel'][t].data])

def test_collision(env):
    import time
    env.reset()
    start = time.time()
    t = 0
    while True:
        t += 1
        (obs_im, obs_vec), goal, reward, done, env_info = env.step([0.1, 0.4])

        # for k in ('intersection_offroad', 'intersection_otherlane'):
        #     print('{0}: {1}'.format(k, env_info[k]))
        # input('')

        # import IPython; IPython.embed()

        # import matplotlib.pyplot as plt
        # plt.imshow(obs_im)
        # plt.show()

        if t % 10 == 0:
            print('time per step: {0}'.format((time.time() - start) / 10.))
            start = time.time()

        if done:
            t = 0
            # break
            env.reset()

if __name__ == '__main__':
    logging.basicConfig(filename='example.log', level=logging.DEBUG)
    env = RccarGazeboEnv(params={'collision_reward': -1})
    test_collision(env)
    # test_reset(env)
    import IPython; IPython.embed()
