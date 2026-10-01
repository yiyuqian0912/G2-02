"""Constants for dataset-g1024-l128-binary."""

# Dimensions
GLOBAL_SIZE = 1024

# Multi-driver configuration defaults
MIN_DRIVERS = 1
MAX_DRIVERS = 4
DEFAULT_DRIVER_RANGE = [1, 4]

# Logical obstacle counts and primitive leaf capacity
MIN_LOGICAL_OBSTACLES = 4
MAX_LOGICAL_OBSTACLES = 16
# Maximum flattened primitive leaves stored per scene (to accommodate compound recipes)
MAX_OBSTACLES = 48
MIN_OBSTACLE_SIZE = 32.0
MAX_OBSTACLE_SIZE = 192.0

# Obstacle shape IDs and parameter width
OBSTACLE_RECTANGLE = 0
OBSTACLE_ELLIPSE = 1
OBSTACLE_TRIANGLE = 2
OBSTACLE_POLYGON = 3

MAX_POLYGON_VERTICES = 12
PARAM_WIDTH = 32  # Accommodates polygon: [vertex_count, x1, y1, ..., x12, y12, ...]

SHAPE_NAME_TO_ID = {
    "rectangle": OBSTACLE_RECTANGLE,
    "ellipse": OBSTACLE_ELLIPSE,
    "triangle": OBSTACLE_TRIANGLE,
    "polygon": OBSTACLE_POLYGON,
}
SHAPE_ID_TO_NAME = {
    OBSTACLE_RECTANGLE: "rectangle",
    OBSTACLE_ELLIPSE: "ellipse",
    OBSTACLE_TRIANGLE: "triangle",
    OBSTACLE_POLYGON: "polygon",
}

# Default sampling distribution
DEFAULT_SINGLE_PRIMITIVE_PROB = 0.60
DEFAULT_COMPOUND_PROB = 0.40

DEFAULT_PRIMITIVE_FAMILY_PROBS = {
    "rectangle": 0.25,
    "ellipse": 0.15,
    "triangle": 0.15,
    "polygon": 0.45,
}


DEFAULT_COMPOUND_RECIPE_PROBS = {
    "l_shape": 0.18,
    "t_shape": 0.18,
    "u_shape": 0.18,
    "cross": 0.12,
    "dumbbell": 0.12,
    "bent_wall": 0.10,
    "irregular_cluster": 0.12,
}

DEFAULT_RECT_AXIS_ALIGNED_PROB = 0.25

# Raster states
FREE_NO_RESPONSE = 0
FREE_RESPONSE = 1
OBSTACLE = 2

# Palette colors: [R, G, B]
PALETTE_RGB = {
    FREE_NO_RESPONSE: (45, 55, 72),    # #2d3748 Dark Slate
    FREE_RESPONSE: (246, 224, 94),     # #f6e05e Golden Yellow
    OBSTACLE: (229, 62, 62),          # #e53e3e Crimson Red
}
PALETTE_HEX = {
    FREE_NO_RESPONSE: "#2d3748",
    FREE_RESPONSE: "#f6e05e",
    OBSTACLE: "#e53e3e",
}

# Metadata defaults
DEFAULT_MASTER_SEED = 20260923
