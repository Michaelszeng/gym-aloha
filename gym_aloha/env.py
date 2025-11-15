import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose


class AlohaEnv(gym.Env):
    """
    Aloha environment.

    Utilities include (currently only for 'insertion' task):
    - Updates info dictionary with:
        - env/is_grasped_left: boolean indicating if left gripper has grasped its target object
        - env/is_grasped_right: boolean indicating if right gripper has grasped its target object
        - env/is_grasped_both: boolean indicating if both grippers have grasped their targets
        - env/collision_force: float indicating the collision force on the robot
    """

    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        if self.obs_type == "state":
            # Determine object state size based on task
            if self.task == "transfer_cube":
                env_state_size = 7  # cube pose: 3 position + 4 quaternion
            elif self.task == "insertion":
                env_state_size = 14  # peg and socket poses: 7 + 7
            else:
                raise NotImplementedError(self.task)

            self.observation_space = spaces.Dict(
                {
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                    "agent_vel": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                    "env_state": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(env_state_size,),
                        dtype=np.float64,
                    ),
                }
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_state":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                    "agent_vel": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            obs = {
                "agent_pos": raw_obs["qpos"].copy(),
                "agent_vel": raw_obs["qvel"].copy(),
                "env_state": raw_obs["env_state"].copy(),
            }
        elif self.obs_type == "pixels":
            obs = {"top": raw_obs["images"]["top"].copy()}
        elif self.obs_type == "pixels_agent_state":
            obs = {
                "top": raw_obs["images"]["top"].copy(),
                "agent_pos": raw_obs["qpos"].copy(),
                "agent_vel": raw_obs["qvel"].copy(),
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        # Compute environment state metrics (at initial state)
        is_grasped_left, is_grasped_right = self.detect_grasp()
        collision_force = self.compute_robot_collision_force(exclude_object_contacts=True)

        info = {
            "is_success": False,
            "env/is_grasped_left": is_grasped_left,
            "env/is_grasped_right": is_grasped_right,
            "env/is_grasped_both": is_grasped_left and is_grasped_right,
            "env/collision_force": collision_force,
        }
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)

        # Check for task success using robust checker
        is_success = self.check_success()
        # Terminate on success - task is complete when peg touches pin
        terminated = is_success

        # Compute environment state metrics
        is_grasped_left, is_grasped_right = self.detect_grasp()
        collision_force = self.compute_robot_collision_force(exclude_object_contacts=True)

        info = {
            "is_success": is_success,
            "env/collision_force": collision_force,
        }

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass

    def detect_grasp(self, min_force: float = 0.5, max_angle: float = 90.0) -> tuple[bool, bool]:
        """
        Detect if objects are grasped using contact force and direction detection.
        Currently only implemented for insertion task.

        An object is considered grasped when ALL of the following conditions are met for BOTH fingers:
        1. Contact force between finger and object must be at least min_force (default 0.5N)
        2. Angle between gripper opening direction and contact force direction must be ≤ max_angle degrees

        Note: These values are automatically computed and stored in:
        - info["env/is_grasped_left"]: Whether left gripper has grasped its target
        - info["env/is_grasped_right"]: Whether right gripper has grasped its target
        - info["env/is_grasped_both"]: Whether both grippers have grasped their targets

        Args:
            min_force: Minimum contact force required (N), default 0.5
            max_angle: Maximum angle between gripper opening and force direction (degrees), default 90.0

        Returns:
            tuple[bool, bool]: (left_grasped, right_grasped)
                - left_grasped: True if left gripper has grasped its target object
                - right_grasped: True if right gripper has grasped its target object

        Raises:
            NotImplementedError: If called on a task other than insertion
        """
        # Only implemented for insertion task
        if self.task != "insertion":
            raise NotImplementedError(f"detect_grasp is only implemented for 'insertion' task, got '{self.task}'")

        physics = self._env.physics

        def _check_finger_grasp(finger_geom_name: str, object_geom_names: list[str]) -> tuple[bool, float, float, int]:
            """
            Check if a finger has a valid grasp on an object.
            Collects all valid contacts and returns the one with the strongest force.

            Returns:
                tuple[bool, float, float, int]: (is_grasped, contact_force, angle_deg, num_contacts_found)
            """
            # Collect all valid contacts for this finger-object pair
            valid_contacts = []
            total_contacts_checked = 0

            for i in range(physics.data.ncon):
                contact = physics.data.contact[i]
                geom1_name = physics.model.id2name(contact.geom1, "geom")
                geom2_name = physics.model.id2name(contact.geom2, "geom")

                # Check if this contact involves the finger and object
                finger_is_geom1 = geom1_name == finger_geom_name
                finger_is_geom2 = geom2_name == finger_geom_name
                object_is_geom1 = geom1_name in object_geom_names
                object_is_geom2 = geom2_name in object_geom_names

                if not ((finger_is_geom1 and object_is_geom2) or (finger_is_geom2 and object_is_geom1)):
                    continue

                # Found a contact between this finger and object
                total_contacts_checked += 1

                # Get contact force magnitude from efc_force
                efc_address = contact.efc_address
                if efc_address < 0 or efc_address >= physics.data.nefc:
                    continue

                contact_force = 0.0
                num_dims = contact.dim
                for j in range(num_dims):
                    force_idx = efc_address + j
                    if force_idx < physics.data.nefc:
                        contact_force += abs(physics.data.efc_force[force_idx])

                # Check force threshold
                if contact_force < min_force:
                    continue

                # Get contact normal direction (first 3 elements of contact frame)
                # The contact frame is stored as a 3x3 rotation matrix in row-major order
                # The normal is the first row (indices 0, 1, 2)
                contact_normal = contact.frame[:3]

                # Normalize the normal vector
                normal_magnitude = np.linalg.norm(contact_normal)
                if normal_magnitude < 1e-6:
                    continue
                contact_normal = contact_normal / normal_magnitude

                # Determine which direction the normal points (from finger or from object)
                # If finger is geom1, normal points from geom1 to geom2 (finger -> object)
                # If finger is geom2, normal points from geom2 to geom1 (object -> finger), so flip it
                if finger_is_geom2:
                    contact_normal = -contact_normal

                # Get gripper body orientation
                # The gripper opening direction is perpendicular to the fingers (y-axis)
                # We use the main gripper_link body which is the parent of both fingers
                arm_prefix = finger_geom_name.split("/")[0]  # e.g., "vx300s_left" or "vx300s_right"
                gripper_body_name = f"{arm_prefix}/gripper_link"
                gripper_body_id = physics.model.name2id(gripper_body_name, "body")

                # Get gripper body rotation matrix (3x3)
                body_xmat = physics.data.xmat[gripper_body_id].reshape(3, 3)

                # Gripper opening direction is along the y-axis (second column)
                # The fingers slide along the y-axis, so contact forces should be roughly along y
                gripper_opening_dir = body_xmat[:, 1]

                # Calculate angle between contact normal and gripper opening direction
                dot_product = np.clip(np.dot(contact_normal, gripper_opening_dir), -1.0, 1.0)
                angle_rad = np.arccos(np.abs(dot_product))  # Use abs to handle both directions
                angle_deg = np.degrees(angle_rad)

                # Check angle threshold
                if angle_deg <= max_angle:
                    valid_contacts.append((contact_force, angle_deg))

            # If we found any valid contacts, return the one with strongest force
            if valid_contacts:
                best_contact = max(valid_contacts, key=lambda x: x[0])
                return True, best_contact[0], best_contact[1], total_contacts_checked

            return False, 0.0, 0.0, total_contacts_checked

        # Insertion task: left gripper -> socket, right gripper -> peg
        left_object_geoms = ["socket-1", "socket-2", "socket-3", "socket-4"]
        right_object_geom = ["red_peg"]

        left_fingers = ["vx300s_left/10_left_gripper_finger", "vx300s_left/10_right_gripper_finger"]
        right_fingers = ["vx300s_right/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger"]

        # Check if both fingers of left gripper have valid grasp
        left_results = [_check_finger_grasp(finger, left_object_geoms) for finger in left_fingers]
        left_grasped = all(result[0] for result in left_results)

        # Check if both fingers of right gripper have valid grasp
        right_results = [_check_finger_grasp(finger, right_object_geom) for finger in right_fingers]
        right_grasped = all(result[0] for result in right_results)

        # Store detailed grasp info for debugging (attached to the env instance)
        self._last_grasp_info = {
            "left_fingers": left_results,
            "right_fingers": right_results,
        }

        return left_grasped, right_grasped

    def compute_robot_collision_force(self, exclude_object_contacts: bool = True) -> float:
        """
        Compute collision forces on robot, with option to exclude object grasping contacts.
        Currently only implemented for insertion task.

        Uses efc_force which contains actual constraint forces from the physics engine.

        Note: This value (with exclude_object_contacts=True) is automatically computed and
        stored in info["env/collision_force"] during each step() call.

        Args:
            exclude_object_contacts: If True, excludes gripper-object contacts (useful for
                                    penalizing collisions while allowing grasping)

        Returns:
            float: Total collision force magnitude on robot bodies

        Raises:
            NotImplementedError: If called on a task other than insertion
        """
        # Only implemented for insertion task
        if self.task != "insertion":
            raise NotImplementedError(
                f"compute_robot_collision_force is only implemented for 'insertion' task, got '{self.task}'"
            )

        physics = self._env.physics

        # Define gripper geoms (same for all tasks)
        gripper_geoms = [
            "vx300s_left/10_left_gripper_finger",
            "vx300s_left/10_right_gripper_finger",
            "vx300s_right/10_left_gripper_finger",
            "vx300s_right/10_right_gripper_finger",
        ]

        # Insertion task object geoms
        object_geoms = ["red_peg", "socket-1", "socket-2", "socket-3", "socket-4"]

        # Get robot body IDs for filtering
        robot_body_ids = set()
        for body_id in range(physics.model.nbody):
            body_name = physics.model.id2name(body_id, "body")
            if body_name is not None and "vx300s" in body_name:
                robot_body_ids.add(body_id)

        # Sum forces from contacts involving robot
        total_collision_force = 0.0
        for i in range(physics.data.ncon):
            contact = physics.data.contact[i]
            geom1_name = physics.model.id2name(contact.geom1, "geom")
            geom2_name = physics.model.id2name(contact.geom2, "geom")

            # Check if this involves the robot
            body1_id = physics.model.geom_bodyid[contact.geom1]
            body2_id = physics.model.geom_bodyid[contact.geom2]

            is_robot_contact = (body1_id in robot_body_ids) or (body2_id in robot_body_ids)

            if not is_robot_contact:
                continue

            # Optionally skip gripper-object contacts (desired grasping contacts)
            if exclude_object_contacts:
                is_gripper = geom1_name in gripper_geoms or geom2_name in gripper_geoms
                is_object = geom1_name in object_geoms or geom2_name in object_geoms

                if is_gripper and is_object:
                    continue  # Skip grasp contacts

            # Get force for this contact from efc_force
            # contact.efc_address gives the index into efc_force array
            efc_address = contact.efc_address
            if efc_address >= 0 and efc_address < physics.data.nefc:
                # Sum constraint forces for this contact
                # Each contact can have multiple constraint forces (normal + friction)
                contact_force = 0.0

                # The contact frame has up to 6 dimensions (3 for friction cone)
                # We sum the absolute values of all constraint forces for this contact
                # Typically: 1 normal + up to 2 friction dimensions
                num_dims = contact.dim
                for j in range(num_dims):
                    force_idx = efc_address + j
                    if force_idx < physics.data.nefc:
                        contact_force += abs(physics.data.efc_force[force_idx])

                total_collision_force += contact_force

        return total_collision_force

    def check_success(self) -> bool:
        """
        Check if the current task has been successfully completed.

        For insertion task:
        - Success criterion: Peg is touching the pin inside the socket (indicates successful insertion)

        Returns:
            bool: True if task is successfully completed

        Raises:
            NotImplementedError: If called on an unsupported task
        """
        if self.task == "insertion":
            physics = self._env.physics

            # Check for pin contact - verify peg is touching the pin inside socket
            for i in range(physics.data.ncon):
                contact = physics.data.contact[i]
                geom1_name = physics.model.id2name(contact.geom1, "geom")
                geom2_name = physics.model.id2name(contact.geom2, "geom")

                if (geom1_name == "red_peg" and geom2_name == "pin") or (
                    geom1_name == "pin" and geom2_name == "red_peg"
                ):
                    return True

            return False
        elif self.task == "transfer_cube":
            raise NotImplementedError("check_success is not implemented for 'transfer_cube' task")
        else:
            raise NotImplementedError(f"check_success is not implemented for task '{self.task}'")
