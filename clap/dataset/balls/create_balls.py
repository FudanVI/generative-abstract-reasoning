import random
import os
import shutil
import yaml
import argparse
import math
import h5py
import numpy as np
from PIL import Image, ImageColor, ImageOps
import torch
from torch.nn.functional import affine_grid, grid_sample
import pymunk


def get_config(cfg_path='config.yaml'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=cfg_path, type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--shape_path', type=str)
    parser.add_argument('--dataset', default='1_obj', type=str)
    arguments = parser.parse_args()
    with open(arguments.config_path) as f:
        config = yaml.safe_load(f)
    for key, val in arguments.__dict__.items():
        if val is not None:
            config[key] = val
    dataset_name = config['dataset']
    config['output_path'] = os.path.join(config['output_path'], 'balls')
    assert dataset_name in config.keys()
    for key, val in config[dataset_name].items():
        config[key] = val
    arguments.__dict__ = config
    return arguments


class GenerationException(Exception):
    pass


class BouncyBalls:

    def __init__(self, cfg):
        # physical space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        # configures
        self.fps = cfg.fps
        self.dt = 1.0 / self.fps
        self.physics_steps_per_frame = cfg.physics_steps_per_frame
        self.total_step = cfg.seqlen
        self.num_obj = cfg.num_obj
        self.shape_path = cfg.shape_path
        self.obj_mass = cfg.obj_mass
        self.obj_shape = cfg.obj_shape
        self.obj_size = cfg.obj_size
        self.obj_colors = cfg.obj_colors
        self.speed = cfg.speed
        self.image_size = cfg.image_size
        self.gravity_size = cfg.gravity_size
        self.self_rotate = cfg.self_rotate
        self.rotate_v = cfg.rotate_v
        self.disentangle = cfg.disentangle

        # init static scenery (barrier walls, etc.)
        self.prepare_scenery()

        # init dynamic objs and collisions for each sample
        self.balls = []
        self.balls_attributes = []
        self.balls_positions = []
        self.balls_velocities = []
        self.create_balls()
        self.counter = {'step': 0}

    def sample(self, index=None, fix=True):
        # init object states
        self.counter['step'] = 0
        # 0 for not collision, 1 for collision
        # collision = random.choice([0, 1])
        collision = random.choice([1])
        # 0 for no gravity, 1 ~ 4 for different gravity directions
        # gravity = random.choice([0, 1, 2, 3, 4])
        gravity = random.choice([0])
        if index is None:
            self.assign_attributes(collision=collision, gravity=gravity)
        else:
            if fix:
                self.assign_attributes_fix(index)
            else:
                self.assign_attributes_var(index)
        # sample video
        v = np.zeros((self.total_step, self.image_size, self.image_size, 3))
        bg = np.zeros((self.image_size, self.image_size, 3))
        ls = np.zeros((self.total_step, self.num_obj, self.image_size, self.image_size, 3))
        cls = np.array([collision, gravity])
        for frame_idx in range(self.total_step):
            if frame_idx > 0:
                for x in range(self.physics_steps_per_frame):
                    self.counter['step'] += 1
                    self.space.step(self.dt / self.physics_steps_per_frame)
            img, bg, layers = self.draw_objects()
            v[frame_idx] = img
            ls[frame_idx] = layers
        return v, bg, ls, cls

    def draw_batch(self, batch_size, index=None, fix=True):
        pics = np.zeros((batch_size, self.total_step, 3, self.image_size, self.image_size))
        for i in range(batch_size):
            vs = self.sample(index=index, fix=fix)[0]
            pics[i] = np.transpose(vs, axes=(0, 3, 1, 2))
        times = np.expand_dims(np.array([1 / self.fps * i for i in range(self.total_step)]), axis=(0, 2))
        times = np.concatenate([times] * batch_size, axis=0)
        return pics / 255., times

    def prepare_scenery(self):
        static_body = self.space.static_body
        s, w = self.image_size, 1.
        static_lines = [
            pymunk.Segment(static_body, (-w, -w), (-w, s + w), w),
            pymunk.Segment(static_body, (-w, s + w), (s + w, s + w), w),
            pymunk.Segment(static_body, (s + w, s + w), (s + w, -w), w),
            pymunk.Segment(static_body, (s + w, -w), (-w, -w), w),
        ]
        for line in static_lines:
            line.filter = pymunk.ShapeFilter(categories=0b01, mask=0b10)
            line.elasticity = 1.0
            line.friction = 0.0
            line.collision_type = 0
        self.space.add(*static_lines)

    def create_balls(self):
        positions = self.rand_pos(self.num_obj, 4)
        if positions is None:
            raise GenerationException('Fail to generate non-overlap objects in the first frame.')
        for x, y in positions:
            self.create_ball(x, y)

    @staticmethod
    def no_gravity(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, (0, 0), damping, dt)

    def constant_gravity(self, direction=(1., 1.)):
        gravity_size = self.gravity_size

        def func(body, gravity, damping, dt):
            gx = gravity_size * direction[0]
            gy = gravity_size * direction[1]
            pymunk.Body.update_velocity(body, (gx, gy), damping, dt)

        return func

    def repel_gravity(self):
        image_size = self.image_size
        gravity_size = self.gravity_size

        def func(body, gravity, damping, dt):
            x, y = body.position
            gx = x - image_size / 2
            gy = y - image_size / 2
            s = math.sqrt(gx ** 2 + gy ** 2)
            gx = gx / s * gravity_size / math.sqrt(s)
            gy = gy / s * gravity_size / math.sqrt(s)
            pymunk.Body.update_velocity(body, (gx, gy), damping, dt)

        return func

    def attract_gravity(self):
        image_size = self.image_size
        gravity_size = self.gravity_size * 4

        def func(body, gravity, damping, dt):
            x, y = body.position
            gx = image_size / 2 - x
            gy = image_size / 2 - y
            s = math.sqrt(gx ** 2 + gy ** 2)
            gx = gx / s * gravity_size / math.sqrt(s)
            gy = gy / s * gravity_size / math.sqrt(s)
            pymunk.Body.update_velocity(body, (gx, gy), damping, dt)

        return func

    def assign_attributes(self, collision=0, gravity=0):
        positions = self.rand_pos(self.num_obj, 4)
        if positions is None:
            raise GenerationException('Fail to generate non-overlap objects in the first frame.')
        for i in range(self.num_obj):
            color_ind = random.randint(0, len(self.obj_colors) - 1)
            color_rgb = np.array(ImageColor.getrgb(self.obj_colors[color_ind]))
            self.balls_attributes[i] = color_rgb
            self.balls_positions[i][0] = positions[i][0]
            self.balls_positions[i][1] = positions[i][1]
            self.balls[i].body.position = tuple(self.balls_positions[i])
            dx = self.image_size / 2 - self.balls[i].body.position[0]
            dy = self.image_size / 2 - self.balls[i].body.position[1]
            dx = dx / math.sqrt((dx ** 2 + dy ** 2))
            dy = dy / math.sqrt((dx ** 2 + dy ** 2))
            self.balls_velocities[i][0] = self.speed * dx
            self.balls_velocities[i][1] = self.speed * dy
            self.balls[i].body.velocity = tuple(self.balls_velocities[i])
            self.balls[i].body.angle = random.random() * math.pi * 2 if self.self_rotate else 0.0
            min_rv, max_rv = self.rotate_v
            self.balls[i].body.angular_velocity = random.random() * (max_rv - min_rv) + min_rv if self.self_rotate else 0.0
            if collision == 0:
                self.balls[i].filter = pymunk.ShapeFilter(categories=0b10, mask=0b01)
            else:
                self.balls[i].filter = pymunk.ShapeFilter(categories=0b10, mask=0b11)
            if gravity == 0:
                self.balls[i].body.velocity_func = self.no_gravity
            elif gravity == 1:
                self.balls[i].body.velocity_func = self.constant_gravity(direction=(0., 1.))
            elif gravity == 2:
                self.balls[i].body.velocity_func = self.constant_gravity(direction=(0., -1.))
            elif gravity == 3:
                self.balls[i].body.velocity_func = self.constant_gravity(direction=(1., 0.))
            else:
                self.balls[i].body.velocity_func = self.constant_gravity(direction=(-1., 0.))

    def assign_attributes_var(self, index):
        attr = self.disentangle['attr'][index]
        positions = self.rand_pos(self.num_obj, 4)
        if positions is None:
            raise GenerationException('Fail to generate non-overlap objects in the first frame.')
        for i in range(self.num_obj):
            if attr == 'Appear':
                color_ind = random.randint(0, len(self.obj_colors) - 1)
                color_rgb = np.array(ImageColor.getrgb(self.obj_colors[color_ind]))
                self.balls_attributes[i] = color_rgb
            if attr == 'Pos':
                self.balls_positions[i][0] = positions[i][0]
                self.balls_positions[i][1] = positions[i][1]
            self.balls[i].body.position = tuple(self.balls_positions[i])
            dx = self.image_size / 2 - self.balls[i].body.position[0]
            dy = self.image_size / 2 - self.balls[i].body.position[1]
            dx = dx / math.sqrt((dx ** 2 + dy ** 2))
            dy = dy / math.sqrt((dx ** 2 + dy ** 2))
            if attr == 'Pos':
                self.balls_velocities[i][0] = self.speed * dx
                self.balls_velocities[i][1] = self.speed * dy
            self.balls[i].body.velocity = tuple(self.balls_velocities[i])

    def assign_attributes_fix(self, index):
        attr = self.disentangle['attr'][index]
        positions = self.rand_pos(self.num_obj, 4)
        if positions is None:
            raise GenerationException('Fail to generate non-overlap objects in the first frame.')
        for i in range(self.num_obj):
            if attr != 'Appear':
                color_ind = random.randint(0, len(self.obj_colors) - 1)
                color_rgb = np.array(ImageColor.getrgb(self.obj_colors[color_ind]))
                self.balls_attributes[i] = color_rgb
            if attr != 'Pos':
                self.balls_positions[i][0] = positions[i][0]
                self.balls_positions[i][1] = positions[i][1]
            self.balls[i].body.position = tuple(self.balls_positions[i])
            dx = self.image_size / 2 - self.balls[i].body.position[0]
            dy = self.image_size / 2 - self.balls[i].body.position[1]
            dx = dx / math.sqrt((dx ** 2 + dy ** 2))
            dy = dy / math.sqrt((dx ** 2 + dy ** 2))
            if attr != 'Pos':
                self.balls_velocities[i][0] = self.speed * dx
                self.balls_velocities[i][1] = self.speed * dy
            self.balls[i].body.velocity = tuple(self.balls_velocities[i])

    def create_ball(self, x, y):
        inertia = pymunk.moment_for_circle(self.obj_mass, 0, self.obj_size, (0, 0))
        body = pymunk.Body(self.obj_mass, inertia)
        body.position = x, y
        body.angle = random.random() * math.pi * 2 if self.self_rotate else 0.0
        min_rv, max_rv = self.rotate_v
        body.angular_velocity = random.random() * (max_rv - min_rv) + min_rv if self.self_rotate else 0.0
        theta = random.random() * math.pi * 2
        vx, vy = self.speed * math.cos(theta), self.speed * math.sin(theta)
        body.velocity = vx, vy
        shape = pymunk.Circle(body, self.obj_size, (0, 0))
        shape.elasticity = 1.0
        shape.friction = 0.0
        shape.collision_type = len(self.balls) + 1
        color_ind = random.randint(0, len(self.obj_colors) - 1)
        color_rgb = np.array(ImageColor.getrgb(self.obj_colors[color_ind]))
        self.space.add(body, shape)
        self.balls.append(shape)
        self.balls_attributes.append(color_rgb)
        self.balls_positions.append([x, y])
        self.balls_velocities.append([vx, vy])

    def rand_pos(self, num_objs, pad, max_try=100):
        min_r = self.image_size / 2 - pad - self.obj_size * 2
        max_r = self.image_size / 2 - pad - self.obj_size
        for _ in range(max_try):
            positions = []
            success_flag = True
            for i in range(num_objs):
                theta = random.random() * math.pi * 2
                x = math.cos(theta) * (random.random() * (max_r - min_r) + min_r) + self.image_size / 2
                y = math.sin(theta) * (random.random() * (max_r - min_r) + min_r) + self.image_size / 2
                if len(positions) > 0:
                    dists = [math.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in positions]
                    if min(dists) < pad + 2 * self.obj_size:
                        success_flag = False
                        break
                positions.append((x, y))
            if success_flag:
                return positions
        return None

    def draw_objects(self):
        s = self.image_size
        temp_path = os.path.join(self.shape_path, '{}.png'.format(self.obj_shape))
        temp = ImageOps.invert(Image.open(temp_path).convert('RGB'))
        temp = temp.resize((self.obj_size * 2, self.obj_size * 2))
        temp = torch.from_numpy(np.array(temp)).float().permute(2, 0, 1)[None, ...] / 255.
        img = torch.zeros(1, 3, s, s).float()
        bg = torch.zeros(1, 3, s, s).float()
        ls = torch.zeros(1, self.num_obj, 3, s, s).float()
        for index, circle in enumerate(self.balls):
            x, y = circle.body.position
            angle = circle.body.angle
            scale = 2 * self.obj_size / s
            t_x, t_y = - (2 * x / s - 1) / scale, - (2 * y / s - 1) / scale
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            theta = torch.tensor([[
                [cos_a / scale, - sin_a / scale, t_x * cos_a - t_y * sin_a],
                [sin_a / scale, cos_a / scale, t_x * sin_a + t_y * cos_a]
            ]]).float()
            grid = affine_grid(theta, [1, 3, s, s], align_corners=False)
            obj = grid_sample(temp, grid, align_corners=False)
            color = torch.from_numpy(self.balls_attributes[index])
            color = color.reshape(1, 3, 1, 1).expand(-1, -1, s, s)
            obj *= color
            if obj.sum() == 0:
                raise GenerationException
            ls[:, index] = obj
            img += obj
        img = img.squeeze(0).clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
        bg = bg.squeeze(0).clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
        ls = ls.squeeze(0).clamp(0, 255).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        return img, bg, ls


def save_data(p, name, v, bg, ls, cls, fps):
    with h5py.File(os.path.join(p, '{}.hdf5'.format(name)), 'w') as f:
        f.create_dataset('video', data=v)
        f.create_dataset('background', data=bg)
        f.create_dataset('objs', data=ls)
        f.create_dataset('classes', data=cls)
        f.create_dataset('fps', data=fps)


if __name__ == '__main__':
    args = get_config()
    random.seed(args.random_seed)
    generator = BouncyBalls(args)
    path = os.path.join(args.output_path, args.dataset)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for split, size in zip(args.split_name, args.split_size):
        num_sample = 0
        video = np.zeros((size, args.seqlen, args.image_size, args.image_size, 3))
        background = np.zeros((size, args.image_size, args.image_size, 3))
        objs = np.zeros((size, args.seqlen, args.num_obj, args.image_size, args.image_size, 3))
        classes = np.zeros((size, 2))
        while num_sample < size:
            try:
                video_sample, background_sample, objs_sample, class_sample = generator.sample()
            except GenerationException as e:
                print(e)
                continue
            video[num_sample] = video_sample
            background[num_sample] = background_sample
            objs[num_sample] = objs_sample
            classes[num_sample] = class_sample
            num_sample += 1
            print('building dataset \'{}\' split \'{}\': {}/{}'.format(args.dataset, split, num_sample, size), end='\r')
        save_data(path, split, video, background, objs, classes, args.fps)
        print()
