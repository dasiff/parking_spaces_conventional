"""
Helper functions for loading polygons and drawing them on images.
"""

import numpy as np
import cv2


def load_polygon(polygon_path):
    """
    Load polygon coordinates from a text file.
    
    Expected format: one coordinate pair per line (x,y)
    Example:
        100,100
        200,100
        200,200
        100,200
    
    Args:
        polygon_path (str): Path to the polygon file
        
    Returns:
        list of tuples: List of (x, y) coordinate pairs, or None if error
    """
    try:
        polygon_coords = []
        with open(polygon_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    # Skip empty lines and comments
                    continue
                
                # Parse coordinate pair
                parts = line.split(',')
                if len(parts) >= 2:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    polygon_coords.append((x, y))
        
        return polygon_coords
    except Exception as e:
        print(f"Error loading polygon from {polygon_path}: {e}")
        return None


def draw_polygon_on_image(image, polygon_coords, color=(0, 255, 0), thickness=2):
    """
    Draw a polygon boundary on an image.
    
    Args:
        image (np.ndarray): Input image (BGR format)
        polygon_coords (list): List of (x, y) coordinate pairs
        color (tuple): BGR color for the polygon boundary (default: green)
        thickness (int): Line thickness in pixels (default: 2)
        
    Returns:
        np.ndarray: Image with polygon drawn on it
    """
    # Create a copy to avoid modifying the original
    result = image.copy()
    
    # Convert coordinates to the format required by cv2.polylines
    pts = np.array(polygon_coords, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw the polygon
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    
    return result


def create_polygon_mask(image_shape, polygon_coords):
    """
    Create a binary mask from polygon coordinates.
    
    Args:
        image_shape (tuple): Shape of the image (height, width) or (height, width, channels)
        polygon_coords (list): List of (x, y) coordinate pairs
        
    Returns:
        np.ndarray: Binary mask where polygon interior is 255, exterior is 0
    """
    # Get height and width from image shape
    if len(image_shape) == 3:
        height, width, _ = image_shape
    else:
        height, width = image_shape
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert coordinates to the format required by cv2.fillPoly
    pts = np.array(polygon_coords, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Fill the polygon
    cv2.fillPoly(mask, [pts], color=255)
    
    return mask
