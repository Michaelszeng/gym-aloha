import mujoco
import mujoco.viewer

from gym_aloha.constants import START_ARM_POSE

model = mujoco.MjModel.from_xml_path("gym_aloha/assets/bimanual_viperx_insertion.xml")

for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    start = model.jnt_qposadr[j]
    size = model.jnt_dofadr[j + 1] - model.jnt_dofadr[j] if j + 1 < model.njnt else model.nq - start
    print(f"{j:2d} {name:25s} qpos[{start}:{start + size}]")

data = mujoco.MjData(model)

data.qpos[: len(START_ARM_POSE)] = START_ARM_POSE
mujoco.mj_forward(model, data)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        # (optionally) update actuator controls here
        viewer.sync()
