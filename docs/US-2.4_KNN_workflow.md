# US-2.4 — Spatiotemporal KNN (from scratch) Exceeds Expectations Workflow

Bu workflow, Issue [#66](https://github.com/hertie-dsa-26/project-chaggg/issues/66) için **from-scratch KNN + KD-tree** implementasyonunu rubric’te **Exceeds Expectations** seviyesine yaklaştırmak üzere tasarlandı. Adım adım ilerleyip, her adım sonunda **run + kontrol** yapacağız ve bu branch’e **küçük, anlamlı commit**’ler atacağız.

---

## Hedef (Issue Acceptance Criteria)

- `SpatiotemporalKNN` sınıfı: `train(data)` ve `predict(location, time)`
- Training: DataFrame → feature vektörleri, **KD-tree** kur
- Prediction: K nearest neighbors → **inverse-distance weighted average** ile crime count
- Split: **2015–2022 train**, **2023–2024 test**
- Test: “all community areas × all months” için prediction üret
- Metrics: **MAE**, **RMSE**
- K tunable: **K=5, 10, 20** dene; en iyisini dokümante et
- Output: CSV veya DB’ye kaydet

---

## Exceeds Expectations kriterleri (rubric mapping)

- **Architecture & Design**
  - `src/algorithms/` altında modüler: `kdtree.py`, `knn_scratch.py`, `metrics.py`, `data_prep.py`
  - CLI runner: `scripts/run_knn_forecast.py` (pipeline + çıktı üretimi)
  - Flask entegrasyonu: `/viz/knn` sayfası + API endpoint; “EDA → model → sonuç” akışı

- **Implementation Quality**
  - KD-tree query \(O(\log n)\) ortalama; brute force fallback opsiyonu (debug)
  - Mesafe metrikleri net (space vs time scaling); edge-case handling (zero distance, missing coords)
  - Büyük data için `usecols`, aggregation ile “monthly counts by community_area” gibi küçültülmüş training table

- **Development Process**
  - Küçük commit’ler, anlamlı mesajlar
  - Minimal ama gerçek testler: KD-tree correctness + KNN weighting + split logic

- **Final Product**
  - Flask’ta user input: `K`, `year range`, `month` seçimi; sonuç harita/tablo

---

## Adım adım workflow (bizim ilerleme planımız)

### Adım 1 — Repo yapısı + skeleton (commit)
**Amaç:** Algoritma kodlarını düzgün yere koymak, import’ları netleştirmek.

- `src/algorithms/` altında:
  - `kdtree.py` (KDNode + build + query)
  - `knn_scratch.py` (`SpatiotemporalKNN`)
  - `metrics.py` (MAE/RMSE)
  - `data_prep.py` (aggregation + split)
- `docs/US-2.4_KNN_report.md` taslağı (ne yaptık, nasıl çalıştırılır)

**Kontrol:** `uv run python -c "from src.algorithms.knn_scratch import SpatiotemporalKNN; print('ok')"`

---

### Adım 2 — Data prep: “community_area × month” target tablosu (commit)
**Amaç:** 8M satırlık incident-level datayı KNN için yönetilebilir tabloya indir.

- Load only needed cols: `date`, `community_area`, `latitude`, `longitude`
- `date` → month (örn. `YYYY-MM`)
- Her community_area için aylık:
  - `crime_count` (target)
  - `centroid_lat/lon` (community_area içindeki olayların ortalaması veya sabit CA centroid)

**Kontrol:** küçük sample ile shape + örnek satırlar.

---

### Adım 3 — Train/test split (commit)
**Amaç:** 2015–2022 / 2023–2024 ayrımını deterministik yap.

- Train months: 2015-01 … 2022-12
- Test months: 2023-01 … 2024-12

**Kontrol:** min/max date; satır sayıları; overlap yok.

---

### Adım 4 — Feature engineering & scaling (commit)
**Amaç:** Spatiotemporal distance’ı “anlamlı” hale getirmek.

Öneri feature vektörü:
- \(x =\) projected meters (UTM16N) veya lat/lon + scale
- \(t =\) month index (örn. 2015-01 = 0)

Distance:
\[
d = \\sqrt{(\\Delta x)^2 + (\\Delta y)^2 + (\\lambda \\Delta t)^2}
\]
Burada \(\lambda\) “1 ay kaç metreye eşit?” ölçeği. \(\lambda\) için 2–3 değer dene ve raporda gerekçelendir.

**Kontrol:** distance fonksiyonu unit test (simetrik, zero distance).

---

### Adım 5 — KD-tree from scratch (commit)
**Amaç:** KD-tree build + kNN query (k neighbors).

- Build: median split (axis alternation)
- Query: backtracking + “best k” listesi (size k max-heap veya sorted list)
- Edge cases: k > n, duplicate points

**Kontrol:** brute force ile random küçük dataset’te aynı neighbor setini veriyor mu (test).

---

### Adım 6 — SpatiotemporalKNN.train/predict (commit)
**Amaç:** Issue API’sini tam karşıla.

- `train(df)`:
  - feature matrix + target vector
  - KD-tree build
- `predict(location, time)`:
  - KD-tree query
  - inverse-distance weights:
    - Eğer distance=0 ise direkt exact neighbor value döndür
    - Aksi halde \(w_i = 1/(d_i + \\epsilon)\)

**Kontrol:** 2-3 küçük senaryo ile beklenen sonucu veriyor mu.

---

### Adım 7 — Batch prediction (all CA × all months) (commit)
**Amaç:** Acceptance’ta istenen “all areas × all months” üretimi.

- Test set tablosu üzerinden her satır için prediction
- Output: `outputs/knn/predictions_knn.csv` (gitignore altında)

**Kontrol:** dosya yazıldı mı, satır sayısı doğru mu.

---

### Adım 8 — Metrics + K tuning (commit)
**Amaç:** K=5,10,20 için MAE/RMSE hesapla ve “best K” raporla.

- `metrics.json` veya `knn_results.csv` (K, MAE, RMSE)
- Rapor dosyasında kısa tablo + yorum

**Kontrol:** metrikler numeric ve beklenen aralıkta.

---

### Adım 9 — Flask entegrasyonu (interactive) (commit)
**Amaç:** “Final Product” kısmını Exceeds’e taşımak.

- `/viz/knn`:
  - input: K (5/10/20), month picker, maybe “space vs time weight”
  - output: table + simple map overlay (CA bazlı)
- `/api/knn/predict?...`:
  - JSON response

**Kontrol:** test client ile 200 + JSON schema kontrol.

---

### Adım 10 — Report + demo script (commit)
**Amaç:** PR reviewer ve hoca “1 dosyadan” anlasın.

- `docs/US-2.4_KNN_report.md`:
  - how to run (scripts + flask)
  - design decisions (KD-tree, scaling, weights)
  - results (K tuning, metrics)
  - limitations

---

## PR açmadan önce son checklist

- `uv run python scripts/run_knn_forecast.py` (smoke run) başarılı
- Flask `/viz/knn` açılıyor, input değişince sonuç değişiyor
- Unit testler çalışıyor (`uv run pytest` varsa)
- `git status` temiz; küçük commit’ler var; branch pushlandı

