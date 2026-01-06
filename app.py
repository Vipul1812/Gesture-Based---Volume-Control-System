from flask import Flask, render_template, Response, jsonify
import cv2
import time
import screen_brightness_control as sbc
import pyautogui
from finger_counter import FingerCounter

app = Flask(__name__)

counter = FingerCounter()
camera = None

prev_vol_dist = None
prev_bri_dist = None

gesture_state = {
    "gesture": "NONE",
    "volume": 0,
    "brightness": 0,
    "quality": "POOR"
}


def draw_status(frame, gesture, distance):
    cv2.rectangle(frame, (20, 20), (420, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"GESTURE : {gesture}",
                (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"DISTANCE : {distance} cm",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)


def gen_frames():
    global camera, prev_vol_dist, prev_bri_dist, gesture_state

    if camera is None:
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, actions = counter.count_fingers(frame)

        # Volume control
        if actions["volume"] is not None:
            if prev_vol_dist is not None:
                diff = actions["volume"] - prev_vol_dist
                if abs(diff) > 0.01:
                    pyautogui.press("volumeup" if diff > 0 else "volumedown")
            prev_vol_dist = actions["volume"]
            gesture_state["volume"] = min(int(actions["volume"] * 100), 100)

        # Brightness control
        if actions["brightness"] is not None:
            if prev_bri_dist is not None:
                diff = actions["brightness"] - prev_bri_dist
                if abs(diff) > 0.01:
                    curr = sbc.get_brightness()[0]
                    new_bri = int(min(max(curr + diff * 120, 0), 100))
                    sbc.set_brightness(new_bri)
            prev_bri_dist = actions["brightness"]
            gesture_state["brightness"] = min(int(actions["brightness"] * 100), 100)

        gesture_state["gesture"] = actions["gesture"]

        if actions["distance_cm"] > 20:
            gesture_state["quality"] = "GOOD"
        elif actions["distance_cm"] > 10:
            gesture_state["quality"] = "MEDIUM"
        else:
            gesture_state["quality"] = "POOR"

        draw_status(frame, actions["gesture"], actions["distance_cm"])

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

        time.sleep(0.02)


# ---------- ROUTES ----------

@app.route("/")
def landing():
    return render_template("index.html")


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gesture_data")
def gesture_data():
    return jsonify(gesture_state)


@app.route("/status")
def status():
    return jsonify({"status": "RUNNING"})


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
