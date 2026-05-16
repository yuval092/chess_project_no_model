# Environment

Board orientation: White starts on rank 1 at minimum Y. File `a` is minimum X and file `h` is maximum X. A square center in world space is `x = board_origin_x + file * square_size + square_size / 2`, `y = board_origin_y + rank * square_size + square_size / 2`, and `z = board_origin_z + board_thickness`.

XML generation uses a local-coordinate rule for geoms inside `board_frame`: square geoms are local offsets relative to the board body, while piece bodies and motion targets use world coordinates. The 64 colored squares are visual-only geoms on top of one continuous collision surface named `board_surface`.

Pieces use primitive collision geometry: a cylinder base and capsule shaft. STL meshes are optional visual-only attachments and are disabled by default with `stl_scale: 0`.

Current arm status: Fetch assets are copied from `gymnasium_robotics/envs/assets/fetch/` into `assets/robot/fetch/`, with STL meshes in `assets/robot/stls/fetch/` and textures in `assets/robot/textures/`. The generator imports the real Fetch body hierarchy, renames `robot0:mocap` to `ee_target`, welds it to `robot0:gripper_link`, measures the end effector at site `robot0:grip`, and uses the real finger joints `robot0:l_gripper_finger_joint` and `robot0:r_gripper_finger_joint`.

Current measured blocker: the real Fetch model loads and has no startup contact with board or pieces, but reachability does not pass for all chess targets. Do not treat waypoint/grasp milestones as physically validated until `mujoco-chess-debug check-reachability --level 1` and `--level 2` pass.
