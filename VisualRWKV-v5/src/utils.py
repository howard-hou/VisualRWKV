import json, time, random, os
import numpy as np
import dataclasses
from torch.nn import functional as F
from typing import List, Dict

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


def crop_6_squares(img, resolution):
    '''Crop 6 squares of resolution from the image. 
       short side is resized to 2 * resolution
       output 6 images
    '''
    # Get the original width and height
    width, height = img.size

    # Determine the shorter side and resize it to 2 * resolution while maintaining aspect ratio
    shorter_side = min(width, height)
    if shorter_side == width:
        new_width = 2 * resolution
        ratio = new_width / width
        new_height = int(height * ratio)
    else:
        new_height = 2 * resolution
        ratio = new_height / height
        new_width = int(width * ratio)
    img = img.resize((new_width, new_height))

    # Crop 6 squares of resolution from the image
    square_size = resolution
    cropped_images = []
    # if the image is a vertical rectangle, crop the top, middle, bottom
    if new_height > new_width:
        for y in [0, (new_height - square_size) // 2, new_height - square_size]:
            for x in [0, new_width - square_size]:
                cropped_images.append(img.crop((x, y, x + square_size, y + square_size)))
    # if the image is a horizontal rectangle, crop the left, middle, right
    else:
        for y in [0, new_height - square_size]:
            for x in [0, (new_width - square_size) // 2, new_width - square_size]:
                cropped_images.append(img.crop((x, y, x + square_size, y + square_size)))
    return cropped_images


def crop_3_squares(img, resolution):
    '''Crop 3 squares of resolution from the image. 
       short side is resized to resolution
       output 3 images
    '''
    # Get the original width and height
    width, height = img.size

    # Determine the shorter side and resize it to resolution while maintaining aspect ratio
    shorter_side = min(width, height)
    if shorter_side == width:
        new_width = resolution
        ratio = new_width / width
        new_height = int(height * ratio)
    else:
        new_height = resolution
        ratio = new_height / height
        new_width = int(width * ratio)
    img = img.resize((new_width, new_height))

    # Crop 3 squares of resolution from the image
    square_size = resolution
    cropped_images = []
    # if the image is a vertical rectangle, crop the top, middle, bottom
    if new_height > new_width:
        for y in [0, (new_height - square_size) // 2, new_height - square_size]:
            cropped_images.append(img.crop((0, y, square_size, y + square_size)))
    # if the image is a horizontal rectangle, crop the left, middle, right
    else:
        for x in [0, (new_width - square_size) // 2, new_width - square_size]:
            cropped_images.append(img.crop((x, 0, x + square_size, square_size)))
    return cropped_images



def crop_4_squares(img, resolution):
    '''Crop 4 squares of resolution from the image. 
       long side is resized to 2 * resolution
       output 4 images of top left, top right, bottom left, bottom right
    '''
    # Get the original width and height
    width, height = img.size

    # Determine the longer side and resize it to 2 * resolution while maintaining aspect ratio
    longer_side = max(width, height)
    if longer_side == width:
        new_width = 2 * resolution
        ratio = new_width / width
        new_height = int(height * ratio)
    else:
        new_height = 2 * resolution
        ratio = new_height / height
        new_width = int(width * ratio)
    img = img.resize((new_width, new_height))

    # Crop 4 squares of resolution from the image
    square_size = resolution
    cropped_images = []
    # from top left, top right, bottom left, bottom right
    for y in [0, new_height - square_size]:
        for x in [0, new_width - square_size]:
            cropped_images.append(img.crop((x, y, x + square_size, y + square_size)))
    return cropped_images


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    id: str
    roles: List[str]
    conversations: List[Dict[str, str]]

    def append_message(self, role, message):
        d = {"from": role, "value": message}
        self.conversations.append(d)