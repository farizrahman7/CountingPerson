import cv2
from ultralytics import YOLO

# URL RTSP dari kamera
rtsp_url = 'rtsp://username:password@ip_address:port/path'

# Memuat model YOLOv5
model = YOLO("yolov5s.pt")  # Menggunakan model YOLOv5 kecil untuk kecepatan, bisa memilih model lain

# Membuka stream RTSP
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Tidak dapat membuka stream RTSP")
    exit()

while True:
    # Membaca frame dari stream
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Tidak dapat membaca frame dari stream")
        break

    # Mengubah format frame menjadi RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi objek dalam frame
    results = model(frame)

    # Filter hasil deteksi untuk objek 'person'
    persons = [result for result in results.pandas().xyxy[0] if result['class'] == 0]

    # Menghitung jumlah orang yang terdeteksi
    jumlah_orang = len(persons)

    # Menampilkan jumlah orang yang terdeteksi
    print(f"Jumlah orang yang terdeteksi: {jumlah_orang}")

    # Menggambar bounding box di sekitar objek yang terdeteksi
    for person in persons:
        bbox = person['bbox']
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, "person", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Menampilkan frame dengan bounding box
    cv2.imshow("Deteksi Orang", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan resource
cap.release()
cv2.destroyAllWindows()