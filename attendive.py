# attentive.py
import cv2
import time
import os
import math
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Try MediaPipe
USE_MEDIAPIPE = True
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    USE_MEDIAPIPE = True
except Exception:
    USE_MEDIAPIPE = False

# Config
SECOND_ATTENTIVE_RATIO = 0.5
EAR_THRESH = 0.22
HEAD_CENTER_MARGIN = 0.18
REPORTS_DIR_DEFAULT = "reports"
os.makedirs(REPORTS_DIR_DEFAULT, exist_ok=True)


def calculate_EAR(eye_points):
    (p1, p2, p3, p4, p5, p6) = eye_points
    A = math.dist(p2, p6)
    B = math.dist(p3, p5)
    C = math.dist(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def generate_overall_dashboard(session_id, per_second_percentages, total_seconds, final_score, output_dir):
    folder = os.path.join(output_dir, f"session_{session_id}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "overall_dashboard.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Timeline
    times = list(range(len(per_second_percentages)))
    axes[0, 0].plot(times, per_second_percentages, marker='o', linewidth=1.25)
    axes[0, 0].axhline(final_score, color='red', linestyle='--', label=f"Average {final_score:.2f}%")
    axes[0, 0].set_title("Engagement Over Time (per second)")
    axes[0, 0].set_xlabel("Seconds")
    axes[0, 0].set_ylabel("Engagement (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    # Pie
    attentive_pct = final_score
    inattentive_pct = 100.0 - attentive_pct
    axes[0, 1].pie(
        [attentive_pct, inattentive_pct],
        labels=[f"Attentive ({attentive_pct:.1f}%)", f"Inattentive ({inattentive_pct:.1f}%)"],
        autopct="%1.1f%%",
        startangle=90
    )
    axes[0, 1].set_title("Attentive vs Inattentive")

    # Bar
    axes[1, 0].bar(["Final Score"], [final_score], color="#1976D2")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_ylabel("Percent (%)")
    axes[1, 0].set_title("Final Attentive Score")
    axes[1, 0].text(0, final_score + 1, f"{final_score:.2f}%", ha='center')

    # Summary
    feedback = _generate_feedback(final_score)
    summary = (
        f"Session ID: {session_id}\n"
        f"Total seconds: {total_seconds}\n"
        f"Final Attentive Score: {final_score:.2f}%\n\n"
        f"Feedback:\n{feedback}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.01, 0.99, summary, va='top', fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path, folder


def _generate_feedback(score):
    if score >= 85:
        return "Excellent engagement! Keep up the focus."
    elif score >= 65:
        return "Good engagement. A little more focus will help."
    elif score >= 40:
        return "Average engagement. Attention drifted intermittently."
    else:
        return "Low engagement. Recommend reducing distractions and looking at the camera."


def start_attentiveness(session_minutes=0, show_window=True, camera_index=0, output_dir=REPORTS_DIR_DEFAULT):
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return {}

    if USE_MEDIAPIPE:
        mp_face = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    session_start_ts = time.time()
    per_second_percentages = []
    total_seconds = 0
    current_second = 0
    frames_this_second = 0
    weight_sum_this_second = 0.0
    last_print_second = -1

    print("🎥 Tracking attentiveness... Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_present, eyes_open, head_forward = False, False, False

            if USE_MEDIAPIPE:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_face.process(rgb)
                if results.multi_face_landmarks:
                    face_present = True
                    lm = results.multi_face_landmarks[0].landmark
                    landmarks = [(p.x * w, p.y * h) for p in lm]
                    try:
                        leftEAR = calculate_EAR([landmarks[i] for i in LEFT_EYE_IDX])
                        rightEAR = calculate_EAR([landmarks[i] for i in RIGHT_EYE_IDX])
                        eyes_open = ((leftEAR + rightEAR) / 2.0) > EAR_THRESH
                    except Exception:
                        eyes_open = True
                    try:
                        nose_x, _ = landmarks[1]
                        norm_x = (nose_x / w) - 0.5
                        head_forward = abs(norm_x) <= HEAD_CENTER_MARGIN
                    except Exception:
                        head_forward = True
            else:
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    face_present = True
                    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                    cx = x + fw / 2.0
                    norm_x = (cx / w) - 0.5
                    head_forward = abs(norm_x) <= HEAD_CENTER_MARGIN
                    roi_gray = gray[y:y + fh, x:x + fw]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                    eyes_open = len(eyes) >= 1

            if face_present:
                if eyes_open and head_forward:
                    weight = 1.0
                elif eyes_open and not head_forward:
                    weight = 0.2
                elif (not eyes_open) and head_forward:
                    weight = 0.0
                else:
                    weight = 0.0
            else:
                weight = 0.0

            # if face_present and eyes_open and head_forward:
            #     weight = 1.0  # fully attentive
            # elif face_present and eyes_open and not head_forward:
            #     weight = 0.3  # eyes open but head not centered
            # else:
            #     weight = 0.0  # eyes closed or no face → zero attentiveness


            elapsed = time.time() - session_start_ts
            sec = int(elapsed)
            if sec == current_second:
                frames_this_second += 1
                weight_sum_this_second += weight
            else:
                if frames_this_second > 0:
                    per_sec_pct = (weight_sum_this_second / frames_this_second) * 100.0
                    per_second_percentages.append(round(per_sec_pct, 2))
                    total_seconds += 1
                current_second = sec
                frames_this_second, weight_sum_this_second = 1, weight

            cumulative_pct = float(np.mean(per_second_percentages)) if per_second_percentages else 0.0

            if show_window:
                color = (0, 200, 0) if cumulative_pct >= 50 else (0, 165, 255)
                cv2.putText(frame, f"LIVE : {cumulative_pct:6.2f}%", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Student Engagement", frame)

            if sec != last_print_second:
                last_print_second = sec
                print(f"\r[LIVE] Elapsed {sec:4d}s | LIVE Attentive score : {cumulative_pct:6.2f}%  ",
                      end="", flush=True)

            if show_window and cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[USER] 'q' pressed — stopping session.")
                break
            if session_minutes > 0 and elapsed >= session_minutes * 60:
                print("\n[TIMER] Session time reached; stopping session.")
                break
            time.sleep(0.002)

        if frames_this_second > 0:
            per_sec_pct = (weight_sum_this_second / frames_this_second) * 100.0
            per_second_percentages.append(round(per_sec_pct, 2))
            total_seconds += 1

    except KeyboardInterrupt:
        print("\n[INT] KeyboardInterrupt received — stopping session.")
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        if USE_MEDIAPIPE:
            mp_face.close()

    final_score = float(np.mean(per_second_percentages)) if per_second_percentages else 0.0
    print("\n\nFinal Attentive Score for this session : {:.2f}%".format(final_score))

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_path, folder = generate_overall_dashboard(session_id, per_second_percentages,
                                                       total_seconds, final_score, output_dir)

    feedback = _generate_feedback(final_score)
    print("Dashboard saved at:", dashboard_path)
    print("Feedback:", feedback)

    return {
        "final_score": final_score,
        "total_seconds": total_seconds,
        "per_second_percentages": per_second_percentages,
        "dashboard": dashboard_path,
        "folder": folder,
        "feedback": feedback
    }


# ---------------- Streamlit UI wrapped into main() ----------------
def main():
    st.title("🎯 Attentiveness Tracker")
    st.write("Monitor and analyze student attentiveness using webcam and face-tracking.")

    session_time = st.slider("Select Session Duration (minutes)", 1, 120, 30)
    run_button = st.button("Start Session")

    if run_button:
        results = start_attentiveness(session_minutes=session_time, show_window=True)
        if results:
            st.success(f"✅ Final Score: {results['final_score']:.2f}%")
            st.image(results["dashboard"], caption="Overall Engagement Dashboard", width="stretch")
            st.write("### Feedback")
            st.write(results["feedback"])


if __name__ == "__main__":
    main()
