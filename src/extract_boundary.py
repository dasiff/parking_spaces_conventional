import cv2
import numpy as np
import os

K_SIZE=40

def extract_red_boundary(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lower red range
    lower1 = np.array([0, 150, 150])
    upper1 = np.array([10, 255, 255])

    # Upper red range
    lower2 = np.array([170, 150, 150])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    red_mask = mask1 | mask2
    return red_mask


def thicken(mask):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def close_gaps(mask, k_size=50):
    kernel = np.ones((k_size,k_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    cleaned = np.zeros_like(mask)
    cleaned[labels == largest] = 255
    return cleaned


def fill_inside(boundary_mask):
    # Find contours from the boundary
    contours, _ = cv2.findContours(
        boundary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("No contours found.")

    # Pick the largest contour (main property boundary)
    largest = max(contours, key=cv2.contourArea)

    # Create blank mask
    filled = np.zeros_like(boundary_mask)

    # Fill the contour interior
    cv2.drawContours(filled, [largest], -1, 255, thickness=-1)

    return filled


def extract_property_region(image, image_name="sample"):
    red = extract_red_boundary(image)
    cv2.imwrite(os.path.join("outputs", "01_" + image_name + '_raw.png'), red)
    thick = thicken(red)
    cv2.imwrite(os.path.join("outputs", "02_" + image_name + "_thick.png"), thick)
    closed = close_gaps(thick)
    cv2.imwrite(os.path.join("outputs", "03_" + image_name + "_closed.png"), closed)
    main = keep_largest_component(closed)
    filled = fill_inside(main)
    return filled


def prop_mask(root_path=r"G:\\My Drive\\parking_spaces\\data\\raw_images", image_name="Mta_Company_9400_N_Central_Expy_Dallas_TX_75231_0800_United_States"):
    image_path = os.path.join(root_path, image_name + ".png")
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    property_mask = extract_property_region(image, image_name)
    out_path = os.path.join("outputs", f"04_property_mask_{image_name}.png")
    cv2.imwrite(out_path, property_mask)

    print(f"Saved property mask to {out_path}.")
    return property_mask


if __name__ == "__main__":
    prop_mask()