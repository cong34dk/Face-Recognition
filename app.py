from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load ảnh khuôn mặt và tên từ thư mục "faces"
known_face_encodings = []
known_face_names = []
for image in os.listdir('faces'):
    face_image = face_recognition.load_image_file(os.path.join('faces', image))
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(os.path.splitext(image)[0])

def recognize_faces(frame):
    # Resize khung hình để tăng tốc độ xử lý
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Tìm khuôn mặt và mã hóa khuôn mặt trong khung hình hiện tại
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # So sánh khuôn mặt với danh sách khuôn mặt đã biết
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Tính toán khoảng cách gần nhất tới khuôn mặt đã biết
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

def generate_frames():
    while True:
        # Đọc khung hình từ camera
        success, frame = camera.read()
        if not success:
            break
        else:
            face_locations, face_names = recognize_faces(frame)

            # # Hiển thị tên khuôn mặt lên khung hình
            
            # for (top, right, bottom, left), name in zip(face_locations, face_names):
            #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #     cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations to original size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the rectangle with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Chuyển đổi khung hình sang định dạng JPEG để truyền qua HTTP
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Trả về khung hình dưới dạng chuỗi bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)