"""Open a saved point cloud in an interactive Open3D window (rotate with the mouse).

Usage:
    python view.py ../../cv-data/depth/outputs/000000039769.ply
"""

import sys

import open3d as o3d

if len(sys.argv) != 2:
    sys.exit(__doc__)
o3d.visualization.draw_geometries([o3d.io.read_point_cloud(sys.argv[1])], window_name="Point cloud")
