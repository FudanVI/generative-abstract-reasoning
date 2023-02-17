import numpy as np
from PIL import Image, ImageDraw, ImageOps


class GraphPlotter:
    def __init__(self, graph_type='grid'):
        self.graph_type = graph_type

    def plot(self, imgs, dataset_params, mask,
             border=0, border_size=2, gap_size=30):
        imgs = imgs * 255
        if self.graph_type[:5] == 'chain':
            return self.plot_chain(
                imgs, dataset_params, mask,
                border=border, border_size=border_size, gap_size=gap_size
            )
        elif self.graph_type[:4] == 'grid':
            return self.plot_grid(
                imgs, dataset_params, mask,
                border=border, border_size=border_size, gap_size=gap_size
            )
        else:
            raise NotImplementedError

    def plot_prediction(self, panel, gt, dataset_params, mask,
                        pad_outside=10, border=0, border_size=2, gap_size=30):
        panel_canvas = self.plot(
            panel, dataset_params, mask,
            border=border, border_size=border_size, gap_size=gap_size
        )
        if gt is not None:
            gt_canvas = self.plot(
                gt, dataset_params, mask.new_zeros(mask.size()),
                border=border, border_size=border_size, gap_size=gap_size
            )
            padding = np.ones((pad_outside, panel_canvas.shape[1], 3))
            canvas = np.concatenate((panel_canvas, padding, gt_canvas), axis=0)
        else:
            canvas = panel_canvas
        canvas = np.transpose(canvas, (2, 0, 1))
        return canvas

    @staticmethod
    def plot_grid(imgs, dataset_params, mask,
                  gap_size=30, border=0, border_size=2, margin=10):
        num_img, img_size = imgs.shape[0], imgs.shape[-1]
        num_row, num_col, num_node = dataset_params['num_row'], \
            dataset_params['num_column'], dataset_params['num_node']
        assert num_node == num_img
        width = num_col * (img_size + border_size * 2) + (num_col - 1) * gap_size + margin * 2
        height = num_row * (img_size + border_size * 2) + (num_row - 1) * gap_size + margin * 2
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        image_centers = []
        image_paste_pos = []
        for r in range(num_row):
            for c in range(num_col):
                paste_x = c * (img_size + border_size * 2 + gap_size) + margin
                paste_y = r * (img_size + border_size * 2 + gap_size) + margin
                image_paste_pos.append((paste_x, paste_y))
                center_x = paste_x + (img_size // 2 + border_size)
                center_y = paste_y + (img_size // 2 + border_size)
                image_centers.append((center_x, center_y))
        for r in range(num_row):
            for c in range(num_col):
                node = r * num_col + c
                if c - 1 >= 0:
                    draw.line(
                        (image_centers[node][0], image_centers[node][1],
                         image_centers[node - 1][0], image_centers[node - 1][1]),
                        fill=border, width=border_size
                    )
                if c + 1 < num_col:
                    draw.line(
                        (image_centers[node][0], image_centers[node][1],
                         image_centers[node + 1][0], image_centers[node + 1][1]),
                        fill=border, width=border_size
                    )
                if r - 1 >= 0:
                    draw.line(
                        (image_centers[node][0], image_centers[node][1],
                         image_centers[node - num_col][0], image_centers[node - num_col][1]),
                        fill=border, width=border_size
                    )
                if r + 1 < num_row:
                    draw.line(
                        (image_centers[node][0], image_centers[node][1],
                         image_centers[node + num_col][0], image_centers[node + num_col][1]),
                        fill=border, width=border_size
                    )
        for node in range(num_img):
            if imgs[node].shape[0] > 1:
                img = Image.fromarray(imgs[node].transpose(1, 2, 0).astype(np.uint8), 'RGB')
            else:
                img = Image.fromarray(imgs[node][0].astype(np.uint8)).convert('RGB')
            if mask is not None and mask[node] == 1:
                bordered_img = ImageOps.expand(img, border=border_size, fill='red')
            else:
                bordered_img = ImageOps.expand(img, border=border_size, fill='black')
            canvas.paste(bordered_img, box=image_paste_pos[node])
        return np.array(canvas) / 255.

    @staticmethod
    def plot_chain(imgs, dataset_params, mask,
                   gap_size=30, border=0, border_size=2, margin=10):
        num_img, img_size = imgs.shape[0], imgs.shape[-1]
        num_node = dataset_params['num_node']
        assert num_node == num_img
        width = num_img * (img_size + border_size * 2) + (num_img - 1) * gap_size + margin * 2
        height = img_size + border_size * 2 + margin * 2
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        image_centers = []
        image_paste_pos = []
        for node in range(num_img):
            paste_x = node * (img_size + border_size * 2 + gap_size) + margin
            paste_y = margin
            image_paste_pos.append((paste_x, paste_y))
            center_x = paste_x + (img_size // 2 + border_size)
            center_y = paste_y + (img_size // 2 + border_size)
            image_centers.append((center_x, center_y))
        for node in range(num_img):
            if node - 1 >= 0:
                draw.line(
                    (image_centers[node][0], image_centers[node][1],
                     image_centers[node - 1][0], image_centers[node - 1][1]),
                    fill=(border, border, border), width=border_size
                )
            if node + 1 < num_img:
                draw.line(
                    (image_centers[node][0], image_centers[node][1],
                     image_centers[node + 1][0], image_centers[node + 1][1]),
                    fill=(border, border, border), width=border_size
                )
        for node in range(num_img):
            if imgs[node].shape[0] > 1:
                img = Image.fromarray(imgs[node].transpose(1, 2, 0).astype(np.uint8), 'RGB')
            else:
                img = Image.fromarray(imgs[node][0].astype(np.uint8)).convert('RGB')
            if mask is not None and mask[node] == 1:
                bordered_img = ImageOps.expand(img, border=border_size, fill='red')
            else:
                bordered_img = ImageOps.expand(img, border=border_size, fill='black')
            canvas.paste(bordered_img, box=image_paste_pos[node])
        return np.array(canvas) / 255.
