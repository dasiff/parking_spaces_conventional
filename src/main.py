#!/usr/bin/env python3
"""
Simple script to load an image and polygon boundary and overlay them.
This is an MVP for fast inference, lightweight and script-based.
"""

import os
import sys
import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from mask_property import load_polygon, draw_polygon_on_image


def main():
    """
    Main function to load an image and polygon and overlay the boundary.
    """
    # Default paths (can be modified for actual usage)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    polygon_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'polygons')
    
    # Check if image and polygon files are provided as arguments
    if len(sys.argv) >= 3:
        image_path = sys.argv[1]
        polygon_path = sys.argv[2]
    else:
        # Use default paths - look for any image in data/ and polygon in polygons/
        image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(data_dir) else []
        polygon_files = [f for f in os.listdir(polygon_dir) if f.endswith('.txt')] if os.path.exists(polygon_dir) else []
        
        if not image_files or not polygon_files:
            print("Usage: python main.py <image_path> <polygon_path>")
            print(f"Or place an image in {data_dir}/ and a polygon file in {polygon_dir}/")
            print("\nExample polygon file format (polygon.txt):")
            print("100,100")
            print("200,100")
            print("200,200")
            print("100,200")
            return
        
        image_path = os.path.join(data_dir, image_files[0])
        polygon_path = os.path.join(polygon_dir, polygon_files[0])
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Load the polygon
    print(f"Loading polygon: {polygon_path}")
    polygon_coords = load_polygon(polygon_path)
    
    if polygon_coords is None or len(polygon_coords) < 3:
        print(f"Error: Could not load valid polygon from {polygon_path}")
        return
    
    # Create a Shapely polygon for validation
    polygon = Polygon(polygon_coords)
    print(f"Polygon loaded with {len(polygon_coords)} vertices")
    print(f"Polygon area: {polygon.area:.2f} square pixels")
    
    # Draw the polygon on the image
    result_image = draw_polygon_on_image(image, polygon_coords)
    
    # Display the result
    print("Displaying result...")
    plt.figure(figsize=(12, 8))
    # Convert BGR to RGB for matplotlib
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.imshow(result_rgb)
    plt.title('Image with Polygon Boundary Overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Optionally save the result
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output.png')
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
