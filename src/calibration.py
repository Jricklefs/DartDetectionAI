import cv2
import numpy as np
from ultralytics import YOLO
import math

def line_ellipse_intersection(center, direction, ellipse):
    """Find intersection of line from center with ellipse, accounting for ellipse rotation."""
    (ecx, ecy), (w, h), angle = ellipse
    dx, dy = direction
    
    angle_rad = np.radians(-angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    
    x0 = center[0] - ecx
    y0 = center[1] - ecy
    
    x0_rot = x0 * cos_a - y0 * sin_a
    y0_rot = x0 * sin_a + y0 * cos_a
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a
    
    a = w / 2
    b = h / 2
    
    if a <= 1e-6 or b <= 1e-6:
        return (int(center[0] + dx * 1000), int(center[1] + dy * 1000))
    
    A = (dx_rot / a) ** 2 + (dy_rot / b) ** 2
    B = 2 * (x0_rot * dx_rot / a ** 2 + y0_rot * dy_rot / b ** 2)
    C = (x0_rot / a) ** 2 + (y0_rot / b) ** 2 - 1
    
    disc = B ** 2 - 4 * A * C
    
    if disc < 0 or abs(A) < 1e-12:
        return (int(center[0] + dx * 1000), int(center[1] + dy * 1000))
    
    t1 = (-B + np.sqrt(disc)) / (2 * A)
    t2 = (-B - np.sqrt(disc)) / (2 * A)
    t = float(max(t1, t2))
    
    x_int_rot = x0_rot + t * dx_rot
    y_int_rot = y0_rot + t * dy_rot
    
    angle_rad = np.radians(angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    
    x_int = x_int_rot * cos_a - y_int_rot * sin_a
    y_int = x_int_rot * sin_a + y_int_rot * cos_a
    
    return (int(x_int + ecx), int(y_int + ecy))


# Load model and image
model = YOLO("C:/Users/clawd/DartDetector/models/dartboard1280imgz_int8_openvino_model", task="detect")
image = cv2.imread("C:/Users/clawd/dartboard_test.jpg")
h, w = image.shape[:2]
print(f"Image shape: {w}x{h}")

results = model(image, imgsz=1280, conf=0.5, verbose=False)

# Collect points by class
cal_points = []
cal1_points = []
cal2_points = []
cal3_points = []
bull_points = []
twenty_points = []

for r in results:
    boxes = r.boxes
    print(f"Detected {len(boxes)} objects")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)
        
        if cls == 0:
            twenty_points.append((cx, cy, conf))
        elif cls == 2:
            bull_points.append((cx, cy, conf))
        elif cls == 3:
            cal_points.append((cx, cy, conf))
        elif cls == 4:
            cal1_points.append((cx, cy, conf))
        elif cls == 5:
            cal2_points.append((cx, cy, conf))
        elif cls == 6:
            cal3_points.append((cx, cy, conf))

overlay = image.copy()

# Fit ellipses
ellipses = {}

def fit_and_draw_ellipse(points, name, color, thickness=2):
    if len(points) >= 5:
        pts = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        try:
            ellipse = cv2.fitEllipse(pts)
            ecenter = (int(ellipse[0][0]), int(ellipse[0][1]))
            axes = (int(ellipse[1][0]/2), int(ellipse[1][1]/2))
            angle = ellipse[2]
            cv2.ellipse(overlay, ecenter, axes, angle, 0, 360, color, thickness)
            print(f"{name}: center={ellipse[0]}, axes={ellipse[1]}, angle={ellipse[2]:.1f}")
            return ellipse
        except Exception as e:
            print(f"  Failed: {e}")
    return None

ellipses['outer_double'] = fit_and_draw_ellipse(cal_points, "Outer double (cal)", (0, 0, 255), 3)
ellipses['outer_triple'] = fit_and_draw_ellipse(cal1_points, "Outer triple (cal1)", (0, 255, 0), 3)
ellipses['inner_triple'] = fit_and_draw_ellipse(cal2_points, "Inner triple (cal2)", (0, 200, 0), 2)
ellipses['inner_double'] = fit_and_draw_ellipse(cal3_points, "Inner double (cal3)", (0, 0, 200), 2)

# Get center from bull detection
if bull_points:
    center = (float(np.mean([p[0] for p in bull_points])), float(np.mean([p[1] for p in bull_points])))
    print(f"Center from bull detection: {center}")
else:
    centers = [e[0] for e in ellipses.values() if e is not None]
    if centers:
        center = (np.mean([c[0] for c in centers]), np.mean([c[1] for c in centers]))
    else:
        center = (w/2, h/2)

# Draw bull
if ellipses['outer_triple']:
    base = ellipses['outer_triple']
    (_, _), (bw, bh), bangle = base
    
    bull_ratio = 15.9 / 107.0
    bullseye_ratio = 6.35 / 107.0
    
    bull_w = bw * bull_ratio
    bull_h = bh * bull_ratio
    bullseye_w = bw * bullseye_ratio
    bullseye_h = bh * bullseye_ratio
    
    bull_center = (int(center[0]), int(center[1]))
    print(f"Bull: center={bull_center}, axes=({bull_w/2:.1f}, {bull_h/2:.1f}), angle={bangle:.1f}")
    print(f"Bullseye: center={bull_center}, axes=({bullseye_w/2:.1f}, {bullseye_h/2:.1f}), angle={bangle:.1f}")
    
    # Store bull ellipse for segment line intersection
    bull_ellipse = ((float(center[0]), float(center[1])), (bull_w, bull_h), bangle)
    bullseye_ellipse = ((float(center[0]), float(center[1])), (bullseye_w, bullseye_h), bangle)
    
    cv2.ellipse(overlay, bull_center, (int(bull_w/2), int(bull_h/2)), bangle, 0, 360, (255, 0, 0), 2)
    cv2.ellipse(overlay, bull_center, (int(bullseye_w/2), int(bullseye_h/2)), bangle, 0, 360, (255, 0, 255), -1)
else:
    bull_ellipse = None
    bullseye_ellipse = None

# Draw detected points
for p in cal_points:
    cv2.circle(overlay, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
for p in cal3_points:
    cv2.circle(overlay, (int(p[0]), int(p[1])), 5, (0, 0, 180), -1)
for p in cal1_points:
    cv2.circle(overlay, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
for p in cal2_points:
    cv2.circle(overlay, (int(p[0]), int(p[1])), 5, (0, 180, 0), -1)
for p in bull_points:
    cv2.circle(overlay, (int(p[0]), int(p[1])), 8, (255, 255, 0), -1)
for p in twenty_points:
    cv2.circle(overlay, (int(p[0]), int(p[1])), 10, (255, 255, 0), 2)

cv2.drawMarker(overlay, (int(center[0]), int(center[1])), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

# Group points by angle to find segment boundaries
all_ring_points = cal_points + cal1_points + cal2_points + cal3_points

point_angles = []
for p in all_ring_points:
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    angle = math.atan2(dy, dx)
    point_angles.append(angle)

point_angles.sort()

ANGLE_TOLERANCE = math.radians(5)
segment_angles = []
used = set()

for i, ang in enumerate(point_angles):
    if i in used:
        continue
    cluster = [ang]
    used.add(i)
    for j, other_ang in enumerate(point_angles):
        if j in used:
            continue
        diff = abs(ang - other_ang)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        if diff < ANGLE_TOLERANCE:
            cluster.append(other_ang)
            used.add(j)
    if len(cluster) >= 2:
        sin_sum = sum(math.sin(a) for a in cluster)
        cos_sum = sum(math.cos(a) for a in cluster)
        avg_angle = math.atan2(sin_sum, cos_sum)
        segment_angles.append(avg_angle)

segment_angles.sort()
print(f"\nFound {len(segment_angles)} segment boundary angles")

# Draw segment lines from bull ring to outer double
if ellipses['outer_double'] and len(segment_angles) >= 10:
    outer_ellipse = ellipses['outer_double']
    for angle in segment_angles:
        dx = math.cos(angle)
        dy = math.sin(angle)
        outer_pt = line_ellipse_intersection(center, (dx, dy), outer_ellipse)
        
        # Start from bull ring intersection instead of arbitrary offset
        if bull_ellipse:
            inner_pt = line_ellipse_intersection(center, (dx, dy), bull_ellipse)
        else:
            inner_pt = (int(center[0] + dx * 15), int(center[1] + dy * 15))
        
        cv2.line(overlay, inner_pt, outer_pt, (255, 255, 255), 1)

# === SEGMENT NUMBER PLACEMENT ===
segments = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Place numbers BETWEEN outer_double and outer_triple (the "single" scoring area)
if twenty_points and center and ellipses['outer_double'] and ellipses['outer_triple'] and len(segment_angles) >= 20:
    angle_to_20 = math.atan2(twenty_points[0][1] - center[1], twenty_points[0][0] - center[0])
    print(f"Angle to detected 20: {math.degrees(angle_to_20):.1f} deg")
    
    def normalize_angle(a):
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return a
    
    angle_to_20_norm = normalize_angle(angle_to_20)
    segment_angles_norm = [normalize_angle(a) for a in segment_angles]
    
    sorted_indices = sorted(range(len(segment_angles_norm)), key=lambda i: segment_angles_norm[i])
    sorted_angles = [segment_angles_norm[i] for i in sorted_indices]
    
    segment_20_index = 0
    for i in range(len(sorted_angles)):
        a1 = sorted_angles[i]
        a2 = sorted_angles[(i + 1) % len(sorted_angles)]
        
        if a2 < a1:
            if angle_to_20_norm >= a1 or angle_to_20_norm < a2:
                segment_20_index = i
                break
        else:
            if a1 <= angle_to_20_norm < a2:
                segment_20_index = i
                break
    
    print(f"Segment 20 is at boundary index: {segment_20_index}")
    
    # Create text ellipse BETWEEN outer_double and outer_triple
    # This is the "single" scoring area where numbers should go
    outer_double = ellipses['outer_double']
    outer_triple = ellipses['outer_triple']
    
    # Midpoint between outer_double and outer_triple
    text_w = (outer_double[1][0] + outer_triple[1][0]) / 2
    text_h = (outer_double[1][1] + outer_triple[1][1]) / 2
    text_angle = outer_double[2]
    text_center = ((outer_double[0][0] + outer_triple[0][0]) / 2, 
                   (outer_double[0][1] + outer_triple[0][1]) / 2)
    
    text_ellipse = (text_center, (text_w, text_h), text_angle)
    print(f"Text ellipse (between outer_double and outer_triple): axes=({text_w:.1f}, {text_h:.1f})")
    
    for i, seg in enumerate(segments):
        boundary_index = (segment_20_index + i) % len(sorted_angles)
        
        a1 = sorted_angles[boundary_index]
        a2 = sorted_angles[(boundary_index + 1) % len(sorted_angles)]
        
        if a2 < a1:
            a2 += 2 * math.pi
        mid_angle = (a1 + a2) / 2
        if mid_angle >= 2 * math.pi:
            mid_angle -= 2 * math.pi
        
        if mid_angle > math.pi:
            mid_angle -= 2 * math.pi
        
        dx = math.cos(mid_angle)
        dy = math.sin(mid_angle)
        text_pt = line_ellipse_intersection(center, (dx, dy), text_ellipse)
        
        text_str = str(seg)
        (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(overlay, text_str, (text_pt[0] - tw//2, text_pt[1] + th//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

cv2.imwrite("C:/Users/clawd/dartboard_result.jpg", overlay)
print("\nSaved to C:/Users/clawd/dartboard_result.jpg")
