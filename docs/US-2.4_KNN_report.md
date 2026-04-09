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

---

## Current state

- KD-tree ve KNN skeleton + scaling var.
- Monthly aggregation + split parquet üretimi var.
- Unit testlerle KD-tree correctness ve exact-match KNN davranışı kanıtlandı.

## Next planned step

- `scripts/run_knn_forecast.py`: train parquet’i okuyup KD-tree ile batch prediction yapma, K=5/10/20 deneme, MAE/RMSE hesaplama, outputs yazma.

