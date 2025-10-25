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

(visualizing-worlds-quiz)=
# Visualizing Worlds Quiz

This page provides a self-check quiz for the tutorial: [](visualizing-worlds).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "Which tool can visualize a world within a notebook?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "RayTracer", "correct": True},
        {"answer": "RViz2", "correct": False},
        {"answer": "URDFParser", "correct": False},
        {"answer": "WorldState", "correct": False}
      ],
    },
    {
      "question": "What is required to use VizMarkerPublisher for visualization?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "An active ROS2 node and spinning thread", "correct": True},
        {"answer": "A GPU-enabled Plotly backend", "correct": False},
        {"answer": "A YAML calibration file", "correct": False},
        {"answer": "A launched Gazebo world", "correct": False}
      ],
    },
    {
      "question": "Which statement about RayTracer is true?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "It is convenient for snapshot visualizations in notebooks but not recommended continuous visualization during runtime.", "correct": True},
        {"answer": "It publishes ROS topics for RViz2 by default.", "correct": False},
        {"answer": "It parses URDF files.", "correct": False},
        {"answer": "It manages world validation.", "correct": False}
      ],
    },
    {
      "question": "How do you properly shut down ROS2 in the RViz2-based approach?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Destroy the node and call rclpy.shutdown()", "correct": True},
        {"answer": "Only call rclpy.shutdown()", "correct": False},
        {"answer": "Only destroy the node", "correct": False},
        {"answer": "Close the RViz2 window; it shuts down automatically", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
