# US-2.4 — Spatiotemporal KNN (from scratch) Report (Living Doc)

Bu doküman **yaşayan rapor**: US-2.4 çalışması boyunca her adım sonunda “ne yaptık / nasıl test ettik / hangi commit” bilgisi buraya eklenecek.

Issue: [US-2.4: KNN Prediction Algorithm #66](https://github.com/hertie-dsa-26/project-chaggg/issues/66)

---

## Scope (Acceptance Criteria)

- `SpatiotemporalKNN` class: `train(data)` + `predict(location, time)`
- Training: DataFrame → feature vectors → **KD-tree build (from scratch)**
- Prediction: kNN → **inverse-distance weighted average** of crime counts
- Train/test split: **2015–2022 train**, **2023–2024 test**
- Predictions: all test community areas × all months
- Metrics: **MAE**, **RMSE**
- K tuning: K ∈ {5, 10, 20}
- Output: CSV/DB

---

## Progress log

### Step 0 — Workflow plan created

- **What**: Exceeds Expectations odaklı adım adım workflow oluşturuldu.
- **Where**: `docs/US-2.4_KNN_workflow.md`
- **Commit**: `b482a97`

### Step 1 — KNN skeleton modules (architecture + scaffolding)

- **What**
  - `src/algorithms/metrics.py`: `mae`, `rmse`
  - `src/algorithms/kdtree.py`: KD-tree build/query scaffolding (median split + backtracking)
  - `src/algorithms/data_prep.py`: incident → monthly community area aggregation helpers
  - `src/algorithms/knn_scratch.py`: `SpatiotemporalKNN` skeleton (`train`/`predict`)
- **Tests / checks**
  - Import + mini `train()`/`predict()` smoke test run (basic numeric output ok)
- **Commit**: `c6d714b`

### Step 2 — Monthly community area prep runner (data-at-scale handling)

- **What**
  - `scripts/run_knn_prep.py`: cleaned data’dan yalnız gerekli sütunlarla (`date`, `community_area`, `latitude`, `longitude`) yükleyip
    `community_area × month` seviyesinde `crime_count`, `lat/lon` proxy ve `month_index` üreten runner.
- **Commit**: `9718e67`

### Step 3 — Deterministic train/test split outputs (2015–2022 vs 2023–2024)

- **What**
  - `scripts/run_knn_prep.py --write-split`: monthly tabloyu ayrıca iki parçaya yazıyor:
    - `monthly_ca_train_2015_2022.parquet`
    - `monthly_ca_test_2023_2024.parquet`
- **Run evidence**
  - monthly total rows: 22,223
  - train rows: 7,392
  - test rows: 1,848
- **Commit**: `46e06d9`

### Step 4 — Space/time scaling + KD-tree correctness tests (Exceeds-quality proof)

- **What**
  - `src/algorithms/distance.py`: `SpaceTimeScaler` (space=`degrees|meters` + `time_scale`)
  - `src/algorithms/knn_scratch.py`: `SpatiotemporalKNN` artık scaler ile feature vektörü üretiyor
  - `tests/test_kdtree_knn.py`:
    - KD-tree kNN sonucu brute-force ile aynı neighbor set (random küçük dataset)
    - exact-match predict test
  - Flask test suite hızlandırma:
    - `src/flask_app/data.py`: `CHAGGG_SKIP_DATA_LOAD=1` ile data load skip
    - `tests/test_flask_routes.py`: test setup’ta env var set ederek hızlı test
    - `src/flask_app/__init__.py`: dashboard routes tekrar eklendi (testler 404 olmasın)
- **Tests / checks**
  - `uv run python -m unittest discover -s tests -p "test_*.py" -q` → **OK** (3 tests, hızlı)
- **Commit**: `61937cb`

### Step 5 — Batch forecast runner + K tuning (US-2.4 Adım 7–8)

- **What**
  - `scripts/run_knn_forecast.py`: train/test parquet’ten `SpatiotemporalKNN` ile tüm test satırlarında tahmin; **MAE/RMSE**; CSV + `forecast_metrics.json`.
- **Default run (space=meters, time_scale=0)** — özet:
  - K=5: MAE **45.08**, RMSE **71.15**
  - K=10: MAE **43.23**, RMSE **67.81**
  - K=20: MAE **42.29**, RMSE **65.68** ← en iyi K
- **Outputs**: `outputs/knn/forecast_predictions_k{5,10,20}.csv`, `outputs/knn/forecast_metrics.json`

### Step 6 — time_scale grid search (workflow “λ” deneysel)

- **What**: `scripts/run_knn_tune_timescale.py` — `time_scale` ∈ {0, 500, 1000, 2000, 5000, 10000} × K ∈ {5, 10, 20}, space=meters.
- **Sonuç (RMSE’ye göre en iyi)**: **`time_scale=0`**, **K=20** (aynı baseline). Pozitif `time_scale` ile zaman boyutu metre ölçeğinde mesafeyi baskıladığı için bu gridde metrikler kötüleşti; farklı ölçek aralıkları veya `space=degrees` ayrıca denenebilir.
- **Output**: `outputs/knn/forecast_metrics_timescale_sweep.json`

### Step 7 — Flask entegrasyonu (US-2.4 Adım 9)

- **What**
  - `/viz/knn`: K ve test ayı seçimi; tablo + Leaflet haritada CA proxy noktaları (CSV’den).
  - `/api/knn/predict`: `community_area`, `month`, isteğe bağlı `k`, `time_scale` — train parquet üzerinde model eğitip tek nokta tahmini (JSON).
- **Önkoşul**: `outputs/knn/` altında prep + forecast çalıştırılmış olmalı; yoksa sayfa talimat gösterir, API 503 döner.
- **Tests**: `tests/test_flask_routes.py` — `/viz/knn` 200; KNN çıktıları varken API smoke.

---

## How to run (özet)

```bash
# 1) Cleaned data sonrası monthly + split
uv run python scripts/run_knn_prep.py --write-split

# 2) Forecast + metrikler
uv run python scripts/run_knn_forecast.py --k 5,10,20 --space meters --time-scale 0

# 3) (İsteğe bağlı) time_scale taraması
uv run python scripts/run_knn_tune_timescale.py

# 4) Web
# CHAGGG_SKIP_DATA_LOAD=1 uv run flask --app src.flask_app:create_app run
```
*(Flask entrypoint projede nasıl tanımlandıysa ona göre uyarlayın.)*

**Tek komut demo (workflow Adım 10):** cleaned data hazırsa repo kökünden:

```bash
./scripts/demo_knn.sh
```

Bu script `run_knn_prep.py --write-split` ve `run_knn_forecast.py` çalıştırır. Forecast sonunda `outputs/knn/predictions_knn.csv`, o koşuda **RMSE’ye göre en iyi K**’nın tahmin dosyasının kopyasıdır (workflow Adım 7’deki isimlendirme).

---

## Limitations

- **Lokasyon proxy**: CA×ay için lat/lon, o hücredeki olayların ortalaması; gerçek community polygon centroid değil.
- **API / portal drift**: Ham veri snapshot’ı farklıysa satır sayıları ve dolayısıyla agregasyon biraz değişebilir.
- **time_scale**: Denenen pozitif grid (`run_knn_tune_timescale.py`) default `space=meters` altında metrikleri kötüleştirdi; daha ince tarama veya `space=degrees` ayrı deney gerektirir.
- **Flask**: Görünüm önceden üretilmiş CSV’lere dayanır; canlı yeniden eğitim `/api/knn/predict` isteği başına yapılır (küçük tablo için ucuz).

---

## Current state

- From-scratch **KD-tree + `SpatiotemporalKNN`**, monthly CA prep, train/test split, batch forecast, K tuning, time_scale grid search ve Flask görünümü/API tamamlandı.
- Üretim dosyaları `outputs/knn/` altında (git’e genelde eklenmez; lokal veya CI artifact).
- **PR öncesi checklist** (`docs/US-2.4_KNN_workflow.md`): smoke forecast (`demo_knn.sh` veya `run_knn_forecast.py`), `/viz/knn`, unit testler, branch push — takım sürecine göre tamamlanacak.

---

## Local branch commit (push yapılmadı)

### `b594f81` — US-2.4 + Exceeds altyapısı (tek entegrasyon commit’i)

- **İçerik (özet):** Flask `/viz/knn` + `/api/knn/predict`, `knn_bruteforce` / `use_bruteforce`, KNN script’leri (`run_knn_forecast`, `run_knn_tune_timescale`, `demo_knn.sh`), GitHub Actions CI, `ARCHITECTURE` / `CONTRIBUTING` / `RUBRIC_MAPPING`, genişletilmiş testler, README + ana sayfa/nav.
- **Kontrol (commit öncesi):**  
  `CHAGGG_SKIP_DATA_LOAD=1 uv run python -m unittest discover -s tests -p "test_*.py" -q` → **OK** (10 test).
- **Push:** Bilinçli olarak **yok** (yalnızca `gasya/us-2.4-knn` üzerinde lokal commit).

### Planlı sıradaki adımlar (her biri için: uygula → kontrol → bu bölüme yaz → onay → devam)

| # | Adım | Durum |
|---|------|--------|
| 1 | Entegrasyon commit’i + unittest yeşili + bu rapor girdisi | **Tamam** (`b594f81`) |
| 2 | `ruff` dev dependency + CI’da `ruff check` / `ruff format --check` (`src/`, `tests/`) | **Tamam** (ruff + unittest yeşil; push yok) |
| 3 | CI’da coverage (`coverage run` + `coverage report`, `fail_under` + DBSCAN omit) | **Tamam** (12 test; rapor ~%80 satır, push yok) |
| 4 | Push + GitHub’da CI yeşili doğrulama | **Onay bekliyor** (push istenince) |

**Not:** Adım 2–4’ü sen “onayla / atla” dedikçe sırayla işleyeceğim.

#### Adım 2 uygulama notu (onay sonrası)

- `dependency-groups.dev` içine **ruff**; `[tool.ruff]` / lint seçimi `pyproject.toml` içinde.
- CI: `uv sync --locked --all-groups` sonra `ruff check src tests`, `ruff format --check src tests` (tüm `scripts/` şimdilik hariç — star-import ve path hack’leri ayrı PR’da temizlenebilir).
- Yerel kontrol: ruff + **10 unittest** yeşil; `ruff format` ile `src/` + `tests/` biçimlendirildi.

#### Adım 3 uygulama notu (onay sonrası)

- Dev dependency **coverage**; `[tool.coverage.run]` / `[tool.coverage.report]` (`source=src`, `omit=dbscan_hotspots`, `fail_under=78`).
- CI: `coverage run -m unittest …` + `coverage report` (eşik aşılırsa job kırmızı).
- `tests/test_metrics.py`: **mae** / **rmse** birim testleri.
- **Push yok.**

