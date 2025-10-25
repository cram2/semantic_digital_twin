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

(world-state-manipulation-quiz)=
# World State Manipulation Quiz

This page provides a self-check quiz for the tutorial: [](world-state-manipulation).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "Which class aggregates the positions/velocities/accelerations/jerks of all DoFs?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "WorldState", "correct": True},
        {"answer": "RayTracer", "correct": False},
        {"answer": "URDFParser", "correct": False},
        {"answer": "ShapeCollection", "correct": False}
      ],
    },
    {
      "question": "How is a free 6DoF connection created in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Connection6DoF(parent, child, _world=world)", "correct": True},
        {"answer": "RevoluteConnection(parent, child)", "correct": False},
        {"answer": "FixedConnection(parent, child)", "correct": False},
        {"answer": "AddBodyConnection(parent, child)", "correct": False}
      ],
    },
    {
      "question": "What context is required when changing the world (e.g., adding bodies or connections)?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "with world.modify_world():", "correct": True},
        {"answer": "with world.lock():", "correct": False},
        {"answer": "world.begin(); ...; world.end();", "correct": False},
        {"answer": "No special context is required", "correct": False}
      ],
    },
    {
      "question": "How can you set the pose of a free connection in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Assign a TransformationMatrix via the origin property", "correct": True},
        {"answer": "Call set_pose(xyzrpy)", "correct": False},
        {"answer": "Modify the child's visual color", "correct": False},
        {"answer": "Use WorldState.set_pose(connection)", "correct": False}
      ],
    },
    {
      "question": "How to query all Drawer views in a world?",
      "type": "many_choice",
      "answer_cols": 2,
      "answers": [
        {"answer": "an(entity(let(Drawer, domain=world.views)))", "correct": True},
        {"answer": "world.get_views_by_type(Drawer)", "correct": True},
        {"answer": "select Drawer from views", "correct": False},
        {"answer": "find_view('Drawer')", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
