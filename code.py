import sys
import cv2
import numpy as np
import os
import time

# Buat direktori untuk menyimpan frame yang diambil
folder_capture = 'captured_frames'
if not os.path.exists(folder_capture):
    os.makedirs(folder_capture)

cap = cv2.VideoCapture('videos/ytt.mp4')

# Setel resolusi kamera menjadi Full HD (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

lebar_min_rect = 60  # lebar minimum persegi panjang
tinggi_min_rect = 60  # tinggi minimum persegi panjang

posisi_garis_hitung = 580
posisi_garis2_hitung = 500

# Koordinat kotak deteksi
kotak_deteksi1 = ((200 + 270, posisi_garis_hitung - 150), (450 + 270, posisi_garis_hitung - 150 + 50))
kotak_deteksi2 = ((220 + 580, posisi_garis2_hitung - 100), (450 + 620, posisi_garis2_hitung - 100 + 50))

def pusat_handel(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

algo = cv2.createBackgroundSubtractorMOG2()

deteksi = []
offset = 6  # toleransi kesalahan antar piksel
offset1 = 4
counter_masuk = 0
counter_keluar = 0
total_terparkir = 0
kapasitas_parkir = 10  # Atur kapasitas parkir di sini

# Inisialisasi frame sebelumnya
ret, frame_sebelumnya = cap.read()
abu_sebelumnya = cv2.cvtColor(frame_sebelumnya, cv2.COLOR_BGR2GRAY)
abu_sebelumnya = cv2.GaussianBlur(abu_sebelumnya, (3, 3), 5)

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    # Konversi ke citra grayscale dan aplikasikan GaussianBlur
    abu = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(abu, (3, 3), 5)

    # Hitung perbedaan antara frame sekarang dan sebelumnya
    diff = cv2.absdiff(abu_sebelumnya, blur)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    abu_sebelumnya = blur

    img_sub = algo.apply(thresh)

    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    counterShape, h = cv2.findContours(
        dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar kotak deteksi
    cv2.rectangle(frame1, kotak_deteksi1[0], kotak_deteksi1[1], (255, 127, 0), 2)
    cv2.rectangle(frame1, kotak_deteksi2[0], kotak_deteksi2[1], (127, 255, 0), 2)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validasi_counter = (w >= lebar_min_rect) and (h >= tinggi_min_rect)
        if not validasi_counter:
            continue

        # Deteksi pusat hanya pada area yang berbeda (berkorelasi dengan pergerakan)
        if cv2.contourArea(c) > 200:
            pusat = pusat_handel(x, y, w, h)
            deteksi.append(pusat)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame1, pusat, 4, (0, 0, 255), -1)

    for (x, y) in deteksi:
        if (kotak_deteksi1[0][0] < x < kotak_deteksi1[1][0]) and (kotak_deteksi1[0][1] < y < kotak_deteksi1[1][1]):
            counter_masuk += 1
            total_terparkir += 1
            cv2.rectangle(frame1, kotak_deteksi1[0], kotak_deteksi1[1], (0, 255, 0), 2)
            # Simpan frame ketika kendaraan masuk
            cv2.imwrite(f'{folder_capture}/kendaraan_masuk_{counter_masuk}.jpg', frame1)
            print("Counter Kendaraan Masuk" + str(counter_masuk))
            if total_terparkir >= kapasitas_parkir:
                cv2.putText(frame1, "Parkir Penuh!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                print("Parkir Penuh!")
        elif (kotak_deteksi2[0][0] < x < kotak_deteksi2[1][0]) and (kotak_deteksi2[0][1] < y < kotak_deteksi2[1][1]):
            counter_keluar += 1
            total_terparkir -= 1
            cv2.rectangle(frame1, kotak_deteksi2[0], kotak_deteksi2[1], (0, 255, 0), 2)
            # Simpan frame ketika kendaraan keluar
            cv2.imwrite(f'{folder_capture}/kendaraan_keluar_{counter_keluar}.jpg', frame1)
            print("Counter Kendaraan Keluar" + str(counter_keluar))

    deteksi = []

    cv2.putText(frame1, "Mobil Masuk: " + str(counter_masuk),
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 3)
    # Ganti warna tulisan "Mobil Keluar" menjadi biru
    cv2.putText(frame1, "Mobil Keluar: " + str(counter_keluar),
                (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.putText(frame1, "Total Parkir: " + str(total_terparkir),
                (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 3)

    cv2.imshow("Video Asli", frame1)

    # Hapus frame yang lebih  dari 24 jam
    files = os.listdir(folder_capture)
    for file in files:
        file_path = os.path.join(folder_capture, file)
        waktu_pembuatan = os.path.getctime(file_path)
        waktu_sekarang = time.time()
        if waktu_sekarang - waktu_pembuatan > 24 * 60 * 60:
            os.remove(file_path)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()