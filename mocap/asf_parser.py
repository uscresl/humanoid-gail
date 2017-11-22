import re, json
import numpy as np
from scipy.linalg import expm

from skeleton import Skeleton


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def sanitize(vec3):
    """ Swaps axes so that the Mocap coordinates fit Mujoco's coordinate system. """
    if len(vec3) != 3:
        return vec3
    return [round(vec3[2], 3), round(vec3[0], 3), round(vec3[1], 3)]


def rot_euler(v, xyz):
    """ Rotate vector v (or array of vectors) by the euler angles xyz. """
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    for theta, axis in zip(xyz, np.eye(3)):
        v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis*-theta)))
    return v


class AsfParser(object):
    def __init__(self):
        self.skeleton = Skeleton()

    def parse(self, file_name):
        """ Loads ASF Mocap file and parses the bone and joint hierarchy. """
        with open(file_name, 'r') as f:
            lines = list(f)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'):
                i += 1
                continue
            if line.startswith(':root'):
                while i < len(lines) - 1:
                    i += 1
                    line = lines[i].strip()
                    if line.startswith(':'):
                        break
                    elif line.startswith('position') or line.startswith('orientation'):
                        split = line.split()
                        self.skeleton.root[split[0]] = [float(x) for x in split[1:]]
            elif line.startswith(':bonedata'):
                bone = {}
                while i < len(lines) - 1:
                    i += 1
                    line = lines[i].strip()
                    split = line.split()
                    if line.startswith(':'):
                        break
                    elif line.startswith('begin'):
                        bone = { 'id': 0 }
                    elif line.startswith('end'):
                        self.skeleton.bones[bone['name']] = bone
                    elif line.startswith('direction') or line.startswith('axis'):
                        bone[split[0]] = np.array([float(x) for x in split[1:] if isfloat(x)])
                    elif line.startswith('length'):
                        bone['length'] = float(split[1])
                    elif line.startswith('name'):
                        bone['name'] = split[1]
                    elif line.startswith('dof'):
                        bone['dof'] = split[1:]
                    elif line.startswith('id'):
                        bone['id'] = int(split[1])
                    elif line.startswith('limits'):
                        line = ' '.join(split[1:])
                        bone['limits'] = []
                        while i < len(lines) and re.search('\((.*?)\)', line):
                            bone['limits'].append([float(x) for x in re.findall('\((.*?)\)', line)[0].split()])
                            i += 1
                            line = lines[i].strip()
                        if len(bone['limits']) > 0:
                            i -= 1
            elif line.startswith(':hierarchy'):
                links = {}
                while i < len(lines) - 1:
                    i += 1
                    line = lines[i].strip()
                    split = line.split()
                    if line.startswith(':') or line.startswith('end'):
                        break
                    elif line.startswith('begin'):
                        pass
                    else:
                        if split[0] == 'root':
                            self.skeleton.hierarchy['root'] = {
                                'name': 'root',
                                'children': {x: self.skeleton.bones[x] for x in split[1:]}
                            }
                            for x in split[1:]:
                                links[x] = self.skeleton.hierarchy['root']['children'][x]
                        else:
                            links[split[0]]['children'] = {x: self.skeleton.bones[x] for x in split[1:]}
                            for x in split[1:]:
                                links[x] = links[split[0]]['children'][x]
            else:
                i += 1

        self.skeleton.compute_aux_matrices()

    def save_json(self, file_name):
        """ Saves the bone hierarchy as JSON file. """
        return json.dump(self.skeleton.hierarchy, open(file_name, 'w'), indent=4)

    def save_mujoco_xml(self, file_name):
        """ Saves the skeleton as Mujoco/Roboschool-compatible XML file. """
        xml = '''<mujoco model="humanoid">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="3" friction="0.8 0.1 0.1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003">
        <!-- <flags solverstat="enable" energy="enable"/>-->
    </option>
    <asset>
        <texture type="skybox" builtin="gradient" width="128" height="128" rgb1=".4 .6 .8" 
            rgb2="0 0 0"/>  
        <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" 
            rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>  
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
            width="512" height="512"/>  

        <material name='MatPlane' reflectance='0.3' texture="texplane" texrepeat="1 1" texuniform="true"/>
        <material name='geom' texture="texgeom" texuniform="true"/>
    </asset>
    <size nkey="5" nuser_geom="1"/>
    <visual>
        <map fogend="5" fogstart="3"/>
    </visual>
    <worldbody>
        <geom name='floor' pos='0 0 0' size='5 5 .125' type='plane' material="MatPlane" condim='3'/>
        <light mode='targetbodycom' target='torso' directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='0 0 4.0' dir='0 0 -1'/>
        <camera name='targeting' pos='1 1 2' mode='targetbodycom' target='torso'/>
        '''

        def traverse(node, level=2):
            snippet = ''
            name = node['name']
            vector = np.array([0, 0, 0])
            if name == 'root':
                snippet += '''<body name="torso" pos="0 0 1.4">\n\t\t\t<freejoint name="root"/>                
                <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
                <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
                <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge"/>\n'''
                level += 1
            else:
                vector = np.array(node['direction']) * node['length'] / 15.
                snippet += '\t'*level + '<body name="{name}" pos="{v[0]} {v[1]} {v[2]}">\n'.format(name=name, v=sanitize(vector))
                level += 1
                # snippet += '\t'*level + ('<geom fromto="{vf[0]} {vf[1]} {vf[2]} {vt[0]} {vt[1]} {vt[2]}" ' +
                #                          'name="{name}" size="0.05" type="capsule" />\n').format(
                #                         **node, vf=sanitize(-vector), vt=[0,0,0])

            if 'axis' not in node:
                euler = np.zeros(3)
            else:
                euler = np.array(node['axis'])*np.pi/180.

            if name != "root":
                snippet += '\t'*level + ('<geom fromto="{vf[0]} {vf[1]} {vf[2]} {vt[0]} {vt[1]} {vt[2]}" ' +
                                         'name="{name}" size="0.05" type="capsule" />\n').format(
                                        **node, vf=sanitize(-vector), vt=[0,0,0])

            snippet += '\t'*level + ('<geom contype="0" pos="{v[0]} {v[1]} {v[2]}" name="vis_{name}" size="0.001" type="sphere" rgba=".2 .5 1 1" />\n').format(
                            **node, v=sanitize(-vector))

            if 'dof' in node:
                colors = {'x': '1 0 0', 'y': '0 1 0', 'z': '0 0 1'}
                dims = 'xyz'
                for i, dof in enumerate(node['dof']):
                    dim = dof[1]
                    dim_index = dims.index(dim)
                    axis = np.zeros(3)
                    axis[dim_index] = 1
                    axis = np.array(sanitize(rot_euler(axis, euler)))
                    snippet += '\t'*level + '<joint '
                    snippet += ('armature="0.1" damping="0.5" name="{name}_{dim}" axis="{axis[0]} {axis[1]} {axis[2]}" '
                                + 'pos="{v[0]} {v[1]} {v[2]}" stiffness="{stiff}" type="hinge" range="{range_l} {range_u}"').format(
                        name=name, dim=dim, axis=axis, v=sanitize(-vector),
                        range_l=node['limits'][i][0], range_u=node['limits'][i][1],
                        stiff=400./node['length']**4.)
                    # print('%s has stiffness %.3f' % (node['name'], 400./node['length']**4.))
                    snippet += ' />\n'
                    # snippet += '\t'*level + ('<geom pos="{v[0]} {v[1]} {v[2]}" name="vis_{name}" size="0.06" type="sphere" rgba=".2 .5 1 1" />\n').format(
                    #                     **node, v=sanitize(-vector))
                    # snippet += '\t'*level + '<geom fromto="{s[0]} {s[1]} {s[2]} {v[0]} {v[1]} {v[2]}" type="capsule" size="0.01" rgba="{color} 1" />\n'.format(s=sanitize(-vector), v=sanitize(rot_euler(axis, euler)*0.3-vector), color=colors[dim])

            if 'children' in node:
                snippet += '\n'.join([traverse(x, level) for x in node['children'].values()])

            level -= 1
            snippet += '\t'*level + '</body>\n'
            return snippet

        xml += traverse(self.skeleton.hierarchy['root'])

        xml += '''\t</worldbody>
    <tendon>
    </tendon>
    <actuator><!-- this section is not supported, same constants in code -->'''
        for j in self.skeleton.bones.values():
            if "dof" not in j:
                continue
            for dof in j["dof"]:
                dim = dof[1]
                xml += '\n' + '\t'*2 + '<motor gear="100" joint="{name}_{dim}" name="{name}_{dim}"/>'.format(**j, dim=dim)
        xml += '''
    </actuator>
</mujoco>
'''
        with open(file_name, 'w') as f:
            f.write(xml)
