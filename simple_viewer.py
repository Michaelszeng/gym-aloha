"""
Simple MuJoCo viewer that loads the XML directly with no environment wrapper.
This lets you interact with the pure physics dynamics.
"""

import mujoco
import mujoco.viewer

# Load the model directly from XML
model = mujoco.MjModel.from_xml_path("gym_aloha/assets/bimanual_viperx_insertion.xml")
data = mujoco.MjData(model)

# Set initial state from keyframe if available
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

print("Launching simple MuJoCo viewer...")

# Launch the passive viewer (includes built-in control UI)
viewer = mujoco.viewer.launch_passive(model, data)

try:
    while viewer.is_running():
        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync the viewer
        viewer.sync()
except KeyboardInterrupt:
    pass
finally:
    viewer.close()
