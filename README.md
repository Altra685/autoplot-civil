# AutoPlot Civil

Perhitungan dan grafik teknik sipil yang dapat dijalankan langsung di peramban.

Repositori ini memuat sebagian modul dari aplikasi yang lebih lengkap. Modul
lanjutan (penurunan, dinding penahan, daya dukung, hidrologi, SPT, Atterberg)
tidak disertakan di sini.

## Modul

| Modul | Isi |
|---|---|
| Beton | Kuat tekan silinder dengan koreksi rasio h/d (SNI 1974:2011) |
| Saringan | Kurva distribusi butiran, Cu, Cc, klasifikasi USCS & AASHTO |
| Pemadatan | Kurva Proctor, MDD, OMC, garis ZAV |
| Statistik | Rerata, simpangan baku, kekuatan karakteristik, uji kenormalan |
| Satuan | Konversi satuan teknik sipil |

## Menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`.

## Menjalankan tes

```bash
python -m pytest tests -q
```

## Contoh data

`examples/` berisi berkas CSV contoh untuk setiap modul.

## Struktur

```
app.py                     aplikasi Streamlit
src/engineering/           modul perhitungan
src/visualization/         pembuat grafik
tests/                     tes unit
examples/                  data contoh
docs/                      ringkasan rumus
```

## Lisensi

MIT.