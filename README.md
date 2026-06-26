**Animal Vision** adalah sebuah aplikasi berbasis Desktop Graphical User Interface (GUI) yang dibangun menggunakan **Python**, **OpenCV**, dan **Tkinter**. Aplikasi ini mensimulasikan bagaimana 50 spesies hewan yang berbeda (termasuk mamalia, burung, serangga, amfibi, reptil, hingga biota laut dalam) melihat dunia di sekitar mereka secara *real-time* menggunakan input dari webcam.

Aplikasi ini mengimplementasikan berbagai teknik digital image processing (DIP) seperti manipulasi ruang warna (*Color Space Transformation*), pergeseran spektrum warna (UV & Inframerah), distorsi geometris lensa (*fisheye* & pembiasan), pemrosesan *framerate*, dan deteksi kontur tepi untuk mendekati keakuratan ilmiah dari anatomi mata masing-masing hewan.

---

## ✨ Fitur Utama

* **50 Filter Penglihatan Hewan Unique:** Mulai dari dikromasi pada anjing, penglihatan termal pada ular, mata majemuk lebah, hingga penglihatan psikedelik udang mantis.
* **Full Desktop GUI Interface:** Navigasi bersih menggunakan pustaka Tkinter, menghilangkan ketergantungan kontrol berbasis Command Line (CMD).
* **Smooth Mousewheel Scrolling:** Panel menu sebelah kanan mendukung *scrolling* penuh menggunakan roda mouse untuk penjelajahan filter yang cepat.
* **Dynamic Scientific Description:** Kotak deskripsi interaktif di bagian bawah yang memperbarui penjelasan ilmiah mengenai anatomi visual hewan terpilih secara otomatis saat tombol diklik.
* **Visual Active Indicator:** Tombol filter yang sedang aktif akan berubah warna menjadi hijau terang sebagai penanda visual yang jelas.
* **Real-Time Mirroring Processing:** Input webcam otomatis diproses secara horizontal (efek cermin) agar pergerakan terasa lebih natural.

---

## 🛠️ Tech Stack & Prasyarat

Sebelum menjalankan proyek ini, pastikan komputer Anda sudah terinstal Python 3.12,aku coba 3.13 gak bisa dan pustaka berikut:

* **OpenCV (cv2):** Untuk pemrosesan video stream dan manipulasi matriks gambar.
* **NumPy:** Untuk komputasi array dan operasi matematika filter warna.
* **Pillow (PIL):** Untuk menjembatani konversi format gambar OpenCV (BGR) ke dalam GUI Tkinter (RGB).
* **Tkinter:** Pustaka GUI bawaan Python.

---
