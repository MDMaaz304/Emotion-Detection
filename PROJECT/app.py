#pip install flask opencv-python deepface# app.py

from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
from collections import deque

# Initialize Flask app
app = Flask(__name__)

# Start video capture
camera = cv2.VideoCapture(0)

# Use deque to store last N frame emotions
max_frames = 20
emotion_history = deque(maxlen=max_frames)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_emotions = {'happy': 0, 'sad': 0}
            total_faces = 0

            try:
                # Analyze the frame
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # If only one face detected, make it a list
                if not isinstance(results, list):
                    results = [results]

                # Count emotions in this frame
                for result in results:
                    dominant_emotion = result['dominant_emotion']

                    if dominant_emotion in frame_emotions:
                        frame_emotions[dominant_emotion] += 1
                        total_faces += 1

                # Save this frame's emotions into history
                emotion_history.append(frame_emotions)

                # Draw bounding boxes
                # Decide box color based on LIVE happy/sad (for each face separately)
                for result in results:
                    x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                    dominant_emotion = result['dominant_emotion']

                    # Default green for happy, red for sad
                    if dominant_emotion == 'happy':
                        box_color = (0, 255, 0)
                    elif dominant_emotion == 'sad':
                        box_color = (0, 0, 255)
                    else:
                        box_color = (255, 255, 255)  # White for others

                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, box_color, 2)

            except Exception as e:
                print("Error:", e)
                pass

            # Now calculate average emotion percentages
            happy_total = sum(frame['happy'] for frame in emotion_history)
            sad_total = sum(frame['sad'] for frame in emotion_history)
            total = happy_total + sad_total

            if total > 0:
                happy_percent = (happy_total / total) * 100
                sad_percent = (sad_total / total) * 100
            else:
                happy_percent = sad_percent = 0

            # Display text at top
            if happy_percent >= sad_percent:
                overall_color = (0, 255, 0)  # Green
            else:
                overall_color = (0, 0, 255)  # Red

            text = f"Happy: {happy_percent:.1f}% | Sad: {sad_percent:.1f}%"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, overall_color, 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
