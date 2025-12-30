import cv2
import numpy as np
from quickdraw import QuickDrawData
from skimage.metrics import structural_similarity as ssim
import time
import random
import mediapipe as mp

# --- Simple words for drawing game ---
SIMPLE_WORDS = [
    "cat", "dog", "tree", "house", "car", "sun", "moon", "star",
    "apple", "banana", "circle", "square", "triangle", "heart",
    "fish", "bird", "flower", "cloud", "mountain", "boat",
    "airplane", "bicycle", "cup", "spoon", "fork", "pencil",
    "book", "clock", "key", "umbrella", "rainbow", "butterfly"
]

def get_random_word():
    """Get a random simple word from the list."""
    return random.choice(SIMPLE_WORDS)

def get_quickdraw_reference(category):
    """Get a reference drawing from QuickDraw dataset."""
    qd = QuickDrawData()
    if category not in qd.drawing_names:
        raise ValueError(f"Invalid category: {category}")
    return qd.get_drawing(category)

def render_drawing_to_image(drawing, size=(28, 28)):
    """Render QuickDraw drawing to image."""
    img = np.ones((255, 255), dtype=np.uint8) * 255
    for stroke in drawing.strokes:
        for i in range(len(stroke) - 1):
            x1, y1 = stroke[i]
            x2, y2 = stroke[i + 1]
            cv2.line(img, (x1, y1), (x2, y2), color=0, thickness=2)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img_resized

def normalize_drawing(img):
    """Normalize drawing to remove style differences."""
    # Convert to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of the drawing
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) == 0:
        return img
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop to bounding box
    cropped = binary[y_min:y_max+1, x_min:x_max+1]
    
    # Add padding and resize to standard size
    h, w = cropped.shape
    max_dim = max(h, w)
    if max_dim == 0:
        return img
    
    # Create square canvas with padding
    pad = max_dim // 10
    square_size = max_dim + 2 * pad
    square = np.ones((square_size, square_size), dtype=np.uint8) * 255
    
    # Center the drawing
    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Resize to standard size
    normalized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    return normalized

def preprocess_canvas(canvas):
    """Preprocess user canvas drawing."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Normalize the drawing
    normalized = normalize_drawing(thresh)
    
    return normalized

def compare_drawings_ssim(user_img, ref_img):
    """Compare using Structural Similarity Index."""
    user = user_img.astype(np.float32) / 255.0
    ref = ref_img.astype(np.float32) / 255.0
    score, _ = ssim(user, ref, full=True, data_range=1.0)
    return score

def compare_drawings_contour(user_img, ref_img):
    """Compare using contour matching."""
    # Find contours
    user_contours, _ = cv2.findContours(user_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_contours, _ = cv2.findContours(ref_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(user_contours) == 0 or len(ref_contours) == 0:
        return 0.0
    
    # Get largest contours
    user_largest = max(user_contours, key=cv2.contourArea)
    ref_largest = max(ref_contours, key=cv2.contourArea)
    
    # Match shapes
    match_score = cv2.matchShapes(user_largest, ref_largest, cv2.CONTOURS_MATCH_I2, 0)
    
    # Convert to similarity (lower match_score = more similar)
    similarity = 1.0 / (1.0 + match_score)
    return similarity

def compare_drawings_histogram(user_img, ref_img):
    """Compare using histogram correlation."""
    user_hist = cv2.calcHist([user_img], [0], None, [256], [0, 256])
    ref_hist = cv2.calcHist([ref_img], [0], None, [256], [0, 256])
    
    correlation = cv2.compareHist(user_hist, ref_hist, cv2.HISTCMP_CORREL)
    return max(0.0, correlation)

def compare_drawings_hausdorff(user_img, ref_img):
    """Compare using Hausdorff distance for shape matching."""
    user_contours, _ = cv2.findContours(user_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_contours, _ = cv2.findContours(ref_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(user_contours) == 0 or len(ref_contours) == 0:
        return 0.0
    
    user_largest = max(user_contours, key=cv2.contourArea)
    ref_largest = max(ref_contours, key=cv2.contourArea)
    
    # Calculate Hausdorff distance
    match_score = cv2.matchShapes(user_largest, ref_largest, cv2.CONTOURS_MATCH_I3, 0)
    similarity = 1.0 / (1.0 + match_score * 10)  # More strict scaling
    return similarity

def compare_drawings_combined(user_img, ref_img):
    """Combine multiple comparison methods for better accuracy."""
    ssim_score = compare_drawings_ssim(user_img, ref_img)
    contour_score = compare_drawings_contour(user_img, ref_img)
    hist_score = compare_drawings_histogram(user_img, ref_img)
    hausdorff_score = compare_drawings_hausdorff(user_img, ref_img)
    
    # More strict weighted combination - require higher scores
    combined = (0.4 * ssim_score + 0.3 * contour_score + 0.15 * hist_score + 0.15 * hausdorff_score)
    
    # Require at least 2 methods to agree (be above 0.5)
    agreement_count = sum([
        ssim_score > 0.5,
        contour_score > 0.5,
        hist_score > 0.5,
        hausdorff_score > 0.5
    ])
    
    return combined, {
        'ssim': ssim_score,
        'contour': contour_score,
        'histogram': hist_score,
        'hausdorff': hausdorff_score,
        'agreement': agreement_count
    }

def main():
    # Get random word
    category = get_random_word()
    print(f"\n🎨 Draw: {category.upper()}")
    print("=" * 50)
    
    # Get reference drawing
    try:
        drawing = get_quickdraw_reference(category)
        reference_img = render_drawing_to_image(drawing)
        reference_img = normalize_drawing(reference_img)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("\nInstructions:")
    print("  • Show your hand to the camera")
    print("  • Point your index finger to draw in the air")
    print("  • Press 's' to submit your drawing")
    print("  • Press 'c' to clear canvas")
    print("  • Press 'q' to quit")
    print("  • Press 'n' for a new random word\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    time.sleep(2)  # Allow camera to warm up
    
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    prev_center = None
    drawing_active = False
    drawing_points = []  # Track all drawing points for area calculation
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks (optional, for visualization)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Get index finger tip (landmark 8)
                h, w, _ = frame.shape
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                center = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Get index finger MCP (base) to check if finger is extended
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                
                # Check if index finger is extended (tip is above PIP joint)
                finger_extended = index_tip.y < index_pip.y
                
                if finger_extended:
                    # Draw finger tip position on frame
                    cv2.circle(frame, center, 10, (0, 255, 0), -1)
                    cv2.circle(frame, center, 15, (0, 255, 0), 2)
                    
                    # Draw on canvas
                    if prev_center:
                        cv2.line(canvas, prev_center, center, (0, 0, 0), 5)
                        drawing_active = True
                        drawing_points.append(center)
                    
                    prev_center = center
                else:
                    prev_center = None
        else:
            prev_center = None
        
        # Add text overlay
        cv2.putText(frame, f"Draw: {category.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Draw: {category.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        # Display frames
        cv2.imshow("Air Drawing - Webcam Feed", frame)
        cv2.imshow("Air Canvas - Press 's' to Submit", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if drawing_active:
                break
            else:
                print("Please draw something first!")
        elif key == ord('c'):
            canvas[:] = 255
            drawing_active = False
            prev_center = None
            drawing_points = []
            print("Canvas cleared!")
        elif key == ord('n'):
            # Get new random word
            category = get_random_word()
            print(f"\n🎨 New word: {category.upper()}")
            canvas[:] = 255
            drawing_active = False
            prev_center = None
            drawing_points = []
            try:
                drawing = get_quickdraw_reference(category)
                reference_img = render_drawing_to_image(drawing)
                reference_img = normalize_drawing(reference_img)
            except ValueError as e:
                print(f"Error: {e}")
        elif key == ord('q'):
            print("Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    
    # Check if drawing has enough content
    if len(drawing_points) < 10:
        print("\n⚠️  Drawing too short! Please draw more.")
        return
    
    # Calculate drawing area coverage
    if len(drawing_points) > 0:
        points_array = np.array(drawing_points)
        x_range = points_array[:, 0].max() - points_array[:, 0].min()
        y_range = points_array[:, 1].max() - points_array[:, 1].min()
        coverage = (x_range * y_range) / (640 * 480)
        if coverage < 0.01:  # Less than 1% of canvas
            print("\n⚠️  Drawing too small! Please draw larger.")
            return
    
    # Process user drawing
    user_img = preprocess_canvas(canvas)
    
    # Check if user drawing has enough content
    user_pixels = np.sum(user_img == 0)
    if user_pixels < 50:  # Too few black pixels
        print("\n⚠️  Drawing too sparse! Please draw more clearly.")
        return
    
    # Show comparison
    comparison = np.hstack([user_img, reference_img])
    comparison_resized = cv2.resize(comparison, (560, 140))
    
    cv2.putText(comparison_resized, "Your Drawing", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(comparison_resized, "Reference", (290, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imshow("Comparison - Your Drawing vs Reference", comparison_resized)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    # Compare drawings
    score, details = compare_drawings_combined(user_img, reference_img)
    
    print("\n" + "=" * 50)
    print(f"📊 Similarity Score: {score:.2%}")
    print(f"   SSIM: {details['ssim']:.2%}")
    print(f"   Contour: {details['contour']:.2%}")
    print(f"   Histogram: {details['histogram']:.2%}")
    print(f"   Hausdorff: {details['hausdorff']:.2%}")
    print(f"   Methods in agreement: {details['agreement']}/4")
    print("=" * 50)
    
    # Stricter matching criteria
    threshold = 0.55  # Higher threshold
    min_agreement = 2  # Require at least 2 methods to agree
    
    # Both conditions must be met
    if score > threshold and details['agreement'] >= min_agreement:
        print(f"✅ Great job! Your drawing matches '{category}'!")
    else:
        print(f"❌ Not quite right. The word was '{category}'. Try again!")
        if score <= threshold:
            print(f"   (Score {score:.2%} below threshold {threshold:.2%})")
        if details['agreement'] < min_agreement:
            print(f"   (Only {details['agreement']}/4 methods agreed)")
    
    print("\nPress any key to continue or close the window...")

if __name__ == "__main__":
    main()

