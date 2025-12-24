import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from modules.detection import BillboardDetector
    print("BillboardDetector imported successfully")
except ImportError as e:
    print(f"Failed to import BillboardDetector: {e}")

try:
    from modules.segmentation import Segmenter
    print("Segmenter imported successfully")
except ImportError as e:
    print(f"Failed to import Segmenter: {e}")

try:
    from modules.replacement import apply_warp
    print("apply_warp imported successfully")
except ImportError as e:
    print(f"Failed to import apply_warp: {e}")

try:
    from modules.tracking import BaseTracker
    print("BaseTracker imported successfully")
except ImportError as e:
    print(f"Failed to import tracking: {e}")

print("Verification complete.")
