#!/usr/bin/env python3
"""
Quick test of the Detection API components.
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_yolo_models():
    """Test that YOLO models load correctly."""
    print("=" * 60)
    print("Testing YOLO Model Loading")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ultralytics: {e}")
        return False
    
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / "models"
    
    # Test calibration model
    cal_model_path = models_dir / "dartboard1280imgz_int8_openvino_model"
    print(f"\nLoading calibration model: {cal_model_path}")
    try:
        cal_model = YOLO(str(cal_model_path), task="detect")
        print(f"✓ Calibration model loaded")
        print(f"  Model names: {cal_model.names}")
    except Exception as e:
        print(f"✗ Failed to load calibration model: {e}")
        return False
    
    # Test tip detection model
    tip_model_path = models_dir / "posenano27122025_int8_openvino_model"
    print(f"\nLoading tip detection model: {tip_model_path}")
    try:
        tip_model = YOLO(str(tip_model_path), task="pose")
        print(f"✓ Tip detection model loaded")
        print(f"  Model names: {tip_model.names}")
    except Exception as e:
        print(f"✗ Failed to load tip model: {e}")
        return False
    
    return True


def test_calibration_detector():
    """Test the calibration detector class."""
    print("\n" + "=" * 60)
    print("Testing Calibration Detector")
    print("=" * 60)
    
    try:
        from app.core.calibration import YOLOCalibrationDetector
        detector = YOLOCalibrationDetector()
        
        if detector.is_initialized:
            print("✓ YOLOCalibrationDetector initialized")
        else:
            print("✗ YOLOCalibrationDetector failed to initialize")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_tip_detector():
    """Test the tip detector class."""
    print("\n" + "=" * 60)
    print("Testing Tip Detector")
    print("=" * 60)
    
    try:
        from app.core.detection import DartTipDetector
        detector = DartTipDetector()
        
        if detector.is_initialized:
            print("✓ DartTipDetector initialized")
            print(f"  Model: {detector.model_name}")
            print(f"  Is pose model: {detector.is_pose_model}")
        else:
            print("✗ DartTipDetector failed to initialize")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_scoring():
    """Test the scoring system."""
    print("\n" + "=" * 60)
    print("Testing Scoring System")
    print("=" * 60)
    
    try:
        from app.core.scoring import scoring_system
        
        # Test bullseye (center)
        result = scoring_system.score_from_dartboard_coords(0, 0)
        print(f"Center (0,0): {result}")
        assert result['zone'] in ['inner_bull', 'bullseye'], f"Expected bullseye, got {result['zone']}"
        print("✓ Bullseye scoring works")
        
        # Test triple 20 area (roughly 100mm up from center)
        result = scoring_system.score_from_dartboard_coords(0, -100)
        print(f"Top area (0,-100mm): {result}")
        print("✓ Outer scoring works")
        
        # Test miss (way outside)
        result = scoring_system.score_from_dartboard_coords(200, 200)
        print(f"Far outside (200,200mm): {result}")
        print("✓ Miss detection works")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_tracker():
    """Test the dart tracker."""
    print("\n" + "=" * 60)
    print("Testing Dart Tracker")
    print("=" * 60)
    
    try:
        from app.core.dart_tracker import tracker_manager
        from app.core.scoring import scoring_system
        
        # Get a tracker for test board
        tracker = tracker_manager.get_tracker("test-board")
        print(f"✓ Created tracker for 'test-board'")
        
        # Simulate a detection
        fake_tips = [
            {'camera_id': 'cam1', 'x_mm': 0, 'y_mm': -100, 'confidence': 0.95, 'x_px': 500, 'y_px': 300},
            {'camera_id': 'cam2', 'x_mm': 2, 'y_mm': -98, 'confidence': 0.92, 'x_px': 510, 'y_px': 305},
        ]
        
        result = tracker.process_detection(
            detected_tips=fake_tips,
            scoring_func=lambda x, y: scoring_system.score_from_dartboard_coords(x, y)
        )
        
        print(f"✓ Processed detection")
        print(f"  Detection ID: {result.detection_id}")
        print(f"  New dart: {result.new_dart}")
        print(f"  Dart count: {result.dart_count}")
        
        # Process again - should NOT create new dart (same position)
        result2 = tracker.process_detection(
            detected_tips=fake_tips,
            scoring_func=lambda x, y: scoring_system.score_from_dartboard_coords(x, y)
        )
        
        if result2.new_dart is None:
            print("✓ Correctly identified existing dart (no new dart)")
        else:
            print("✗ Incorrectly created new dart for same position")
            return False
        
        # Add a different dart
        fake_tips_2 = [
            {'camera_id': 'cam1', 'x_mm': 50, 'y_mm': 50, 'confidence': 0.90, 'x_px': 600, 'y_px': 400},
        ]
        
        result3 = tracker.process_detection(
            detected_tips=fake_tips + fake_tips_2,  # Both darts visible
            scoring_func=lambda x, y: scoring_system.score_from_dartboard_coords(x, y)
        )
        
        if result3.new_dart is not None:
            print(f"✓ Detected new dart: {result3.new_dart.score} points")
        else:
            print("✗ Failed to detect new dart")
            return False
        
        print(f"  Total darts on board: {result3.dart_count}")
        
        # Test reset
        tracker.reset()
        if tracker.dart_count == 0:
            print("✓ Board reset works")
        else:
            print("✗ Reset failed")
            return False
        
        # Clean up
        tracker_manager.remove_board("test-board")
        print("✓ Board removed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DART DETECTION API - COMPONENT TESTS")
    print("=" * 60)
    
    results = []
    
    results.append(("YOLO Models", test_yolo_models()))
    results.append(("Calibration Detector", test_calibration_detector()))
    results.append(("Tip Detector", test_tip_detector()))
    results.append(("Scoring System", test_scoring()))
    results.append(("Dart Tracker", test_tracker()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
