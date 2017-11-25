# mocap-visualizer
Visualize motion capture data in JavaScript.

**[View Demo](http://eric-heiden.com/mocap-visualizer/)**

## Generate Animation
First, convert absolute joint positions to the animation JSON format that can be played back by this visualization tool.
```bash
python3 generate_animation.py joints_abs.pkl
```

## Publish
A primitive way to see the visualization is by issuing
```bash
python -m SimpleHTTPServer 8000
```
and pointing your browser to `localhost:8000`.
