"""Scripts that run inside Blender (bpy), started as a separate process by executor/render.py:blender_cmd.
They are not imported by the rest of the package.

  executor     render_curves.py      curves.json -> rendered frames
  critics      check_arm_shape.py    arm folds / kinks / sinking into the body, on the posed mesh (no rendering)
               check_arm_width.py    arm width kept along its length (a manual check of the rig)
  evaluation   measure_motion.py     the character's motion, hands and framing, for the metrics
  characters   make_model.py, rig_soft.py, fist.py, make_capsule.py, make_puppet.py
  shared       pose_utils.py (one frame of curves.json -> pose), scene_utils.py (camera, lights, render settings)
"""
