from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
from os import path

from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc

import matplotlib.pyplot as plt
import numpy as np

from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('foldername', None, 'Folder with AMC files to be converted.')

plot_frames = 10
subplot_width = 200
subplot_height = 200


def main(_):
    amc_files = sorted(glob(path.join(FLAGS.foldername, '*.amc')))

    fig, axarr = plt.subplots(len(amc_files), plot_frames)

    for k, filename in enumerate(amc_files):
        env = humanoid_CMU.stand()

        # Parse and convert specified clip.
        converted = parse_amc.convert(filename,
                                      env.physics, env.control_timestep())

        max_frame = converted.qpos.shape[1] - 1
        print('Playing back %i frames.' % max_frame)

        video = np.zeros((plot_frames, subplot_height, subplot_width, 3), dtype=np.uint8)
        frame = 0
        frame_dt = max(int(round(max_frame * 1. / plot_frames)), 1)
        for i in range(max_frame):
            p_i = converted.qpos[:, i]
            with env.physics.reset_context():
                env.physics.data.qpos[:] = p_i
                # obs = env.task.get_observation(env.physics)

            if frame < plot_frames and i % frame_dt == 0:
                video[frame] = env.physics.render(subplot_height, subplot_width, camera_id=1)
                frame += 1

        for i in range(frame):
            axarr[k, i].imshow(video[i])
            axarr[k, i].xaxis.set_visible(False)
            axarr[k, i].yaxis.set_visible(False)
            # if i == frame // 2:
            #     axarr[k, i].set_title(path.basename(filename))

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('foldername')
    app.run(main)
