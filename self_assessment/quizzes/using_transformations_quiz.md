---
jupytext:
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

(using-transformations-quiz)=
# Using Transformations Quiz

This page provides a self-check quiz for the tutorial: [](using-transformations).  
Source: [Jupyter quiz](https://pypi.org/project/jupyterquiz/#description). $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What does the notation `A_T_B` mean?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Transforms frame A to frame B.", "correct": False, "feedback": "Incorrect!" },
        { "answer": "Transforms frame B to frame A.", "correct": True, "feedback": "Correct. `A_T_B` is pose of B in the frame of A." },
        { "answer": "A translation-only offset.", "correct": False, "feedback": "Incorrect. It can include rotation and translation." },
        { "answer": "A pure rotation matrix.", "correct": False, "feedback": "Incorrect. It is a rigid transform (4x4 homogeneous)." }
      ],
    },
    {
      "question": "Which statement about composition is correct?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "`world_T_base @ base_T_camera` yields `world_T_camera`.", "correct": True, "feedback": "Correct! Composition follows the chain of frames."},
        { "answer": "`base_T_camera @ world_T_base` yields `world_T_camera`.", "correct": False, "feedback": "Incorrect order. Left frame must match parent of the right."},
        { "answer": "Composition order does not matter.", "correct": False, "feedback": "Incorrect. Order matters." },
        { "answer": "The result is always expressed in the right-most frame.", "correct": False, "feedback": "Incorrect. It is expressed in the left-most frame." }
      ],
    },
    {
      "question": "Select all true statements about roll, pitch, yaw (RPY).",
      "type": "many_choice",
      "answer_cols": 4,
      "answers": [
        { "answer": "roll: rotation about x-axis", "correct": True },
        { "answer": "pitch: rotation about y-axis", "correct": True },
        { "answer": "yaw: rotation about z-axis", "correct": True },
        { "answer": "angles are specified in degrees in `from_xyz_rpy`", "correct": False },
        { "answer": "RPY order is roll, then pitch, then yaw", "correct": True }
      ],
    },
    {
      "question": "What does the `reference_frame` parameter control in `TransformationMatrix.from_xyz_rpy`?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "The frame in which the translation and rotation are expressed.", "correct": True, "feedback": "Correct! Motions are expressed in the axes of the reference frame." },
        { "answer": "Whether the transform is active or passive.", "correct": False, "feedback": "Incorrect. It does not toggle connection type." },
        { "answer": "Whether units are meters or centimeters.", "correct": False, "feedback": "Incorrect. Units are meters by convention." },
        { "answer": "It only affects rotation, not translation.", "correct": False, "feedback": "Incorrect. It affects both." }
      ],
    },
    {
      "question": "You want to move a camera 0.1 m forward along its OWN x-axis and yaw it by 45°. Which is the most direct approach?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Create `camera_T_offset = TransformationMatrix.from_xyz_rpy(x=0.1, yaw=radians(45), reference_frame=camera_body)` and compose `base_T_camera @ camera_T_offset`.", "correct": True, "feedback": "Correct! Express motion in the local camera frame and compose on the right." },
        { "answer": "Create the offset with `reference_frame=world.root` and multiply on the left.", "correct": False, "feedback": "Incorrect. That would express the move in world axes, not camera axes." },
        { "answer": "Scaling the mesh by 0.1 along x.", "correct": False, "feedback": "Incorrect. Scaling is not a rigid transform of the pose." },
        { "answer": "Change only yaw on the base connection.", "correct": False, "feedback": "Incorrect. That rotates around the base, not the camera's own axis." }
      ],
    },
    {
      "question": "What is the mathematical type of a rigid transform used here?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "4×4 homogeneous transformation matrix", "correct": True, "feedback": "Correct!" },
        { "answer": "3×3 rotation matrix only", "correct": False, "feedback": "Incorrect. That lacks translation." },
        { "answer": "Translation vector only", "correct": False, "feedback": "Incorrect." },
        { "answer": "Quaternion only", "correct": False, "feedback": "Incorrect. Quaternions encode rotation only." }
      ],
    }
  ]

import json
json_str = json.dumps(questions)
json.loads(json_str)

display_quiz(questions)
```
