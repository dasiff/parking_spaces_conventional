# parking_spaces_conventional

Count parking spaces in a parking lot on a property. Takes as input a scale 21 google map image and property boundary, outputs image with numbered parking spaces.

## Project Structure

```
parking_spaces_conventional/
├── src/
│   ├── main.py           # Main script to load image and polygon overlay
│   └── mask_property.py  # Helper functions for polygon operations
├── data/                 # Image data (ignored by git)
├── polygons/             # Polygon definition files
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## Features

- Lightweight and script-based MVP for fast inference
- No web framework, no Docker, no UI, no training code
- Simple image and polygon boundary overlay
- Uses OpenCV for image processing and Shapely for polygon operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dasiff/parking_spaces_conventional.git
cd parking_spaces_conventional
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Script

You can run the script in two ways:

1. **With command-line arguments:**
```bash
python src/main.py <path_to_image> <path_to_polygon>
```

2. **Using default paths:**
Place your image in the `data/` directory and your polygon file in the `polygons/` directory, then run:
```bash
python src/main.py
```

### Polygon File Format

Polygon files should be text files with one coordinate pair per line (x,y format):

```
100,100
200,100
200,200
100,200
```

Lines starting with `#` are treated as comments.

### Example

```bash
# Create sample directories if they don't exist
mkdir -p data polygons

# Place your image and polygon files
cp your_image.jpg data/
cp your_polygon.txt polygons/

# Run the script
python src/main.py
```

## Dependencies

- numpy: Numerical operations
- opencv-python: Image processing
- shapely: Polygon geometry operations
- matplotlib: Visualization

## Output

The script will:
1. Load the specified image and polygon
2. Overlay the polygon boundary on the image in green
3. Display the result using matplotlib
4. Save the output as `output.png` in the project root

## Notes

- This is an MVP focused on fast inference
- The `data/` directory is ignored by git (add your images there)
- Images should be in standard formats (.jpg, .jpeg, .png)
- Polygon coordinates are in pixel coordinates relative to the image
