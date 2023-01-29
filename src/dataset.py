from __future__ import annotations
import torch
import numpy as np
import imageio
from util import load_json
from rich.progress import track
from rich import print


class Dataset():
    # initialized
    root: str
    folder: str

    # loaded for the first image (assuming uniform camera)
    w: int    # note that cx = w/2
    h: int    # note that cy = h/2
    fx: float    # focal length in pixels (x direction)
    fy: float    # focal length in pixels (y direction)

    # loaded for all images
    paths: list[str] = []
    images: torch.Tensor = torch.empty(0)    # shape: (N, 3, H, W)
    poses: torch.Tensor = torch.empty(0)    # shape: (N, 4, 4)

    def __init__(self, root, folder) -> None:
        self.root = root
        self.folder = folder

    def load(self) -> None:
        json = load_json(f"{self.root}/transforms_{self.folder}.json")

        frames = json["frames"]
        if len(frames) == 0:
            raise ValueError("no frames found in dataset")

        # OpenGL -> OpenCV
        # WARNING: not sure if needed
        GL2CV: torch.Tensor = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        image_list_tmp: list[torch.Tensor] = []
        pose_list_tmp: list[torch.Tensor] = []
        for i, frame in enumerate(track(frames, description="loading dataset")):
            # load path
            file_name: str = frame["file_path"].split("/")[-1]
            path: str = f"{self.root}/{self.folder}/{file_name}"
            self.paths.append(path)

            # load pixel
            pixels = imageio.imread(path)
            # WARNING: not sure if needed
            image_list_tmp.append(
                torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0)

            # load pose
            pose: torch.Tensor = torch.tensor(frame["transform_matrix"],
                                              dtype=torch.float32)
            pose = GL2CV @ pose
            pose_list_tmp.append(pose)

            if i == 0:
                self.h, self.w, _ = pixels.shape
                # camera_angle_x = FOV in x direction = 2atan(w/(fl_x*2))
                # camera_angle_y = FOV in y direction = 2atan(h/(fl_y*2))
                # see: https://github.com/NVlabs/instant-ngp/issues/332
                # WARNING: make sure calculation here is correct, fx has pixel unit
                self.fx = float(0.5 * self.w /
                                np.tan(0.5 * json["camera_angle_x"]))
                self.fy = float(0.5 * self.h /
                                np.tan(0.5 * json["camera_angle_y"]))

        # accumulate
        self.images = torch.stack(image_list_tmp, dim=0)
        self.pose = torch.stack(pose_list_tmp, dim=0)

        # check for shape correctness
        print("Loaded dataset:")
        print(f"  - {self.images.shape[0]} frames")
        print(f"  - {self.images.shape[1]} channels")
        print(f"  - {self.images.shape[2]} height")
        print(f"  - {self.images.shape[3]} width")
        print(f"  - {self.h/2} cy")
        print(f"  - {self.w/2} cx")
        print(f"  - {self.fx} fx")
        print(f"  - {self.fy} fy")
        print(f"  - average red: {self.images[:, 0, :, :].mean()}")
        print(f"  - average green: {self.images[:, 1, :, :].mean()}")
        print(f"  - average blue: {self.images[:, 2, :, :].mean()}")
        print(f"  - average alpha: {self.images[:, 3, :, :].mean()}")

        assert (self.images.shape[0] == len(frames))
        assert (self.images.shape[1] == 4)    # RGBA
        assert (self.images.shape[2] == self.h)
        assert (self.images.shape[3] == self.w)

    def generate_rays(self) -> torch.Tensor:
        # generate the origins of the rays in world space
        origins = self.poses[:, None, :3, 3]    # shape: (N, 1, 3)
        origins = origins.expand(-1, self.h * self.w,
                                 -1)    # shape: (N, H*W, 3)

        # generate the directions of the rays in camera space
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
        )    # shape: (H, W)
        xx = (xx - self.w / 2) / self.fx    # shape: (H, W)
        yy = (yy - self.h / 2) / self.fy    # shape: (H, W)
        zz = torch.ones_like(xx)    # shape: (H, W)
        dirs = torch.stack((xx, yy, zz), dim=-1)    # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)    # shape: (H, W, 3)
        del xx, yy, zz

        # transform the directions of the rays to world space
        dirs = torch.einsum(
            "nij,nhwkj->nhwki",
            self.poses[:, :3, :3],    # shape: (N, 3, 3)
            dirs.expand(len(self.images), -1, -1. -1),    # shape: (N, H, W, 3)
        ) # shape: (N, H, W, 3)

        # concatenate the origins and directions of the rays
        dirs = dirs.reshape(len(self.images), -1, 3)    # shape: (N, H*W, 3)
        rays = torch.cat((origins, dirs), dim=-1).contiguous()    # shape: (N, H*W, 6)
        return rays

