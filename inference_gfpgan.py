import argparse
import base64
import cv2
import glob
import numpy as np
import os
import torch
from dataclasses import dataclass
from basicsr.utils import imwrite

from gfpgan import GFPGANer

@dataclass
class Args:
    version: str
    upscale: int
    bg_upsampler: str
    bg_tile: int
    only_center_face: bool
    aligned: bool

async def fix_faces(data, sio, sid):
    """Inference demo for GFPGAN (for users).
    """

    args = Args(
        version="1.3",
        upscale=1,
        bg_upsampler="realesrgan",
        bg_tile=400,
        only_center_face=False,
        aligned=False,
    )

    # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if args.version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
    elif args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('../GFPGAN/experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    # read image
    encoded = data.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded), np.uint8)
    input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=args.aligned, only_center_face=args.only_center_face, paste_back=True)

    # save restored img
    if restored_img is not None:
        success, encoded_image = cv2.imencode(".png", restored_img)
        return encoded_image.tobytes()