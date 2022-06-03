from gym.spaces import Space

class EnvSpec(object):

    def __init__(
            self,
            observation_space,
            observation_im_space,
            observation_scan_space,
            action_space,
            observation_vec_spec,
            action_spec,
            goal_spec):
        """
        :type observation_im_space: Space
        :type observation_scan_space: Space
        :type action_space: Space
        :type observation_vec_spec: dictionary of Space
        :type action_spec: dictionary of Space
        :type goal_spec: dictionary of Space
        """
        self._observation_im_space = observation_im_space
        self._observation_scan_space = observation_scan_space
        self._observation_space = observation_scan_space  # change if needed
        self._action_space = action_space
        self._observation_vec_spec = observation_vec_spec
        self._action_spec = action_spec
        self._goal_spec = goal_spec

    @property
    def observation_im_space(self):
        return self._observation_im_space

    @property
    def observation_scan_space(self):
        return self._observation_scan_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def action_selection_space(self):
        return self._action_selection_space

    @property
    def observation_vec_spec(self):
        return self._observation_vec_spec

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def action_selection_spec(self):
        return self._action_selection_spec

    @property
    def goal_spec(self):
        return self._goal_spec

