# ─────────────────────────────────────────────
#  config.py  —  All tunable constants
# ─────────────────────────────────────────────

# Camera
CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720

# Drawing
STROKE_THICKNESS  = 6       # px
ERASER_RADIUS     = 40      # px  (2-finger gesture eraser)
BTN_ERASER_RADIUS = 50      # px  (toolbar eraser-mode button)

# Gesture engine
SMOOTHING_WINDOW  = 5       # frames for landmark rolling average
DEBOUNCE_FRAMES   = 3       # frames before mode switch is accepted

# Toolbar
TOOLBAR_HEIGHT    = 70      # px  (top bar)
SWATCH_SIZE       = 44      # px  (each color square)
SWATCH_MARGIN     = 10      # px  (left margin + gap between swatches)
BTN_WIDTH         = 110     # px  (action buttons)
BTN_MARGIN        = 12      # px  (gap before/between action buttons)

# Colors  —  BGR format for OpenCV
# (name, bgr_tuple)
PALETTE = [
    ("Blue",   (255,  41,  41)),
    ("Red",    ( 57,  53, 229)),
    ("Green",  ( 71, 160,  67)),
    ("Yellow", (  0, 214, 255)),
    ("Purple", (170,  36, 142)),
    ("Pink",   ( 99,  30, 233)),
    ("Black",  ( 33,  33,  33)),
    ("Grey",   (158, 158, 158)),
]

DEFAULT_COLOR = PALETTE[0][1]   # Blue

# Window
WINDOW_NAME = "Finger Paint"