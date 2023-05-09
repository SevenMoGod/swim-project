import os
import cv2
import torch
import numpy as np


def show_mask(frame, mask, random_color=False):
    if random_color:
        color = np.random.randint(0, 256, size=(3,))
    else:
        color = np.array([0, 255, 0])

    # h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # mask_image = np.clip(mask_image, 0, 255).astype(np.uint8)
    # frame = np.clip(frame, 0, 255).astype(np.uint8)
    # # print(frame.shape, mask_image.shape)
    # frame = cv2.addWeighted(frame, 1, mask_image, 0.6, 0)
    alpha = 0.5
    # Create a copy of the original image and apply the mask
    overlay = frame.copy()
    # overlay = np.uint8(overlay)
    overlay[mask[0] > 0] = color

    # Apply the overlay with alpha blending
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return frame


def show_box_and_label(frame, box, label):
    x1, y1 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    x2, y2 = x1 + w, y1 + h
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame


def save_mask_data(output_dir, mask):
    # cv2.imwrite(os.path.join(output_dir, f"mask_0.jpg"), cv2.cvtColor(np.uint8(np.transpose(mask.cpu().numpy(), (1, 2, 0))), cv2.COLOR_GRAY2BGR))
    # value = 0  # 0 for background

    mask_img = np.zeros(mask.shape[-2:])
    mask_img[mask[0] == True] = 255
    cv2.imwrite(os.path.join(output_dir, 'mask_0.jpg'), mask_img)


if __name__ == "__main__":
    frame = np.ones((1080, 1920, 3))
    mask = np.zeros((1, 1080, 1920))
    print(np.transpose(mask, (1, 2, 0)).squeeze().shape)
    frame = show_mask(frame, mask)

    box = np.array([1, 2, 3, 4])
    label = "person"
    frame = show_box_and_label(frame, box, label)

