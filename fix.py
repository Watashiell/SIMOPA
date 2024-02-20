# Import library yang diperlukan
import sys
import cv2
import numpy as np
import os
import time
from rpi_lcd import LCD  # Import perpustakaan LCD
import threading

# Buat direktori untuk menyimpan frame yang diambil
folder_capture = 'captured_frames'
if not os.path.exists(folder_capture):
    os.makedirs(folder_capture)

# Inisialisasi objek kamera (gunakan kamera dengan ID 0)
cap = cv2.VideoCapture(0)

# Setel resolusi kamera menjadi Full HD (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Tentukan lebar dan tinggi minimum untuk objek yang dideteksi
lebar_min_rect = 60
tinggi_min_rect = 60

# Tentukan posisi garis-garis untuk deteksi kendaraan masuk dan keluar
posisi_garis_hitung = 580
posisi_garis2_hitung = 500

# Koordinat kotak deteksi untuk kendaraan masuk dan keluar
kotak_deteksi1 = ((200 + 270, posisi_garis_hitung - 150), (450 + 270, posisi_garis_hitung - 150 + 50))
kotak_deteksi2 = ((220 + 580, posisi_garis2_hitung - 100), (450 + 620, posisi_garis2_hitung - 100 + 50))

# Fungsi untuk menghitung pusat suatu objek berdasarkan koordinat dan ukurannya
def pusat_handel(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Inisialisasi algoritma pengurangan latar belakang (MOG2)
algo = cv2.createBackgroundSubtractorMOG2()

# Inisialisasi list untuk menyimpan pusat objek yang terdeteksi
deteksi = []

# Toleransi kesalahan antar piksel
offset = 6
offset1 = 4

# Inisialisasi variabel untuk menghitung kendaraan masuk, keluar, dan total terparkir
counter_masuk = 0
counter_keluar = 0
total_terparkir = 0

# Kapasitas parkir maksimum
kapasitas_parkir = 7  

# Inisialisasi frame sebelumnya untuk perbandingan
ret, frame_sebelumnya = cap.read()
abu_sebelumnya = cv2.cvtColor(frame_sebelumnya, cv2.COLOR_BGR2GRAY)
abu_sebelumnya = cv2.GaussianBlur(abu_sebelumnya, (3, 3), 5)

# Inisialisasi LCD
lcd = LCD()

# Fungsi untuk update tampilan LCD menggunakan thread terpisah
def update_lcd():
    while True:
        lcd.text(f"Sisa Parkir: {kapasitas_parkir - total_terparkir}", 1)
        time.sleep(2)
        lcd.text(f"Parkir Terisi: {total_terparkir}", 1)
        time.sleep(2)

# Start thread untuk update LCD
lcd_thread = threading.Thread(target=update_lcd)
lcd_thread.daemon = True
lcd_thread.start()

# Loop utama untuk membaca setiap frame dari kamera
while True:
    # Baca frame dari kamera
    ret, frame1 = cap.read()
    if not ret:
        break

    # Konversi frame ke citra grayscale dan aplikasikan GaussianBlur
    abu = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(abu, (3, 3), 5)

    # Hitung perbedaan antara frame sekarang dan sebelumnya
    diff = cv2.absdiff(abu_sebelumnya, blur)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Update frame sebelumnya dengan frame sekarang
    abu_sebelumnya = blur

    # Aplikasikan algoritma MOG2 untuk mendeteksi perubahan pada frame
    img_sub = algo.apply(thresh)

    # Lakukan operasi morfologi untuk meningkatkan deteksi
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Temukan kontur objek pada frame
    counterShape, h = cv2.findContours(
        dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar kotak deteksi untuk kendaraan masuk dan keluar
    cv2.rectangle(frame1, kotak_deteksi1[0], kotak_deteksi1[1], (255, 127, 0), 2)
    cv2.rectangle(frame1, kotak_deteksi2[0], kotak_deteksi2[1], (127, 255, 0), 2)

    # Loop melalui setiap kontur dan lakukan deteksi objek
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

    # Loop melalui pusat deteksi dan lakukan pengecekan apakah masuk ke area kotak deteksi
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

    # Reset list deteksi setelah satu frame selesai diproses
    deteksi = []

    # Tampilkan informasi jumlah mobil masuk, keluar, dan total parkir pada frame
    cv2.putText(frame1, "Mobil Masuk: " + str(counter_masuk),
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 3)
    # Ganti warna tulisan "Mobil Keluar" menjadi bir
    cv2.putText(frame1, "Mobil Keluar: " + str(counter_keluar),
                (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Tampilkan informasi total parkir pada frame
    cv2.putText(frame1, "Total Parkir: " + str(total_terparkir),
                (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 3)

    # Tampilkan frame dengan overlay deteksi dan informasi parkir
    cv2.imshow("Video Asli", frame1)

    # Hapus frame yang lebih tua dari 24 jam
    files = os.listdir(folder_capture)
    for file in files:
        file_path = os.path.join(folder_capture, file)
        waktu_pembuatan = os.path.getctime(file_path)
        waktu_sekarang = time.time()
        if waktu_sekarang - waktu_pembuatan > 24 * 60 * 60:
            os.remove(file_path)

    # Hentikan loop jika tombol Enter ditekan
    if cv2.waitKey(1) == 13:
        break

# Tutup semua jendela dan lepaskan kamera
cv2.destroyAllWindows()
cap.release()