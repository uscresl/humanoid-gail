import pickle as pkl
import sys, json


def load_pickled_positions(pkl_file):
    with open(pkl_file, "rb") as f:
        traj_data = pkl.load(f, encoding="latin1")
        return traj_data


def rename_joints(positions):
    renaming = {
        'Neck/Nose': 'lowerneck',
        'RElbow': 'rradius' ,
        'Thorax': 'thorax',
        'LWrist': 'lhand',
        'RKnee': 'rtibia',
        'LELbow': 'lradius',
        'Head': 'head',
        'RWrist': 'rhand',
        'LFoot': 'lfoot',
        'RFoot': 'rfoot',
        'LHip': 'lhipjoint',
        'LKnee': 'ltibia',
        'LShoulder': 'lclavicle',
        'RHip': 'rhipjoint',
        'Hip': 'root',
        'RShoulder': 'rclavicle',
        'Spine': 'upperback'
    }
    frames = []
    center = positions[0]["Hip"].copy()
    for f in positions:
        frame = {}
        for key, values in f.items():
            vs = values - center
            vs[0], vs[1], vs[2] = vs[0], vs[2] + 800, vs[1]
            frame[renaming[key]] = vs.tolist()
        frames.append(frame)
    return frames


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 %s INPUT.pkl" % sys.argv[0])
    else:
        positions = load_pickled_positions(sys.argv[1])
        json.dump(
            { "frames": rename_joints(positions) },
            open("./mocap/animation.json", "w"),
            indent=4)
        print("Animation saved successfully.")
