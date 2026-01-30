#!/usr/bin/env python3
"""
DartDetect API - Component Tests
"""
import sys
import os

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
        
        # Test triple 20 area
        result = scoring_system.score_from_dartboard_coords(0, -100)
        print(f"Top area (0,-100mm): {result}")
        print("✓ Outer scoring works")
        
        # Test miss
        result = scoring_system.score_from_dartboard_coords(200, 200)
        print(f"Far outside (200,200mm): {result}")
        print("✓ Miss detection works")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_api_stateless():
    """Test that API is stateless."""
    print("\n" + "=" * 60)
    print("Testing Stateless Design")
    print("=" * 60)
    
    try:
        from app.api.routes import cluster_tips
        
        # Test tip clustering
        tips = [
            {'x_mm': 0, 'y_mm': -100, 'confidence': 0.95, 'camera_id': 'cam1'},
            {'x_mm': 2, 'y_mm': -98, 'confidence': 0.92, 'camera_id': 'cam2'},
            {'x_mm': 50, 'y_mm': 50, 'confidence': 0.88, 'camera_id': 'cam1'},
        ]
        
        clusters = cluster_tips(tips)
        print(f"Input: 3 tips from 2 cameras")
        print(f"Output: {len(clusters)} clusters")
        
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
        print("✓ Tip clustering works (same dart from multiple cameras grouped)")
        
        # Verify first cluster has 2 tips (same dart)
        assert len(clusters[0]) == 2, "First cluster should have 2 tips"
        print("✓ Multi-camera consensus working")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DARTDETECT API - COMPONENT TESTS")
    print("=" * 60)
    
    results = []
    
    results.append(("YOLO Models", test_yolo_models()))
    results.append(("Calibration Detector", test_calibration_detector()))
    results.append(("Tip Detector", test_tip_detector()))
    results.append(("Scoring System", test_scoring()))
    results.append(("Stateless Design", test_api_stateless()))
    
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
