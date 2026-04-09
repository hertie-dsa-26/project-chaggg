# Rubric → evidence (Exceeds hedefi)

Hoca/reviewer için kısa eşleme: hangi rubric maddesi repoda nerede somutlaşıyor.

| Rubric alanı | Exceeds için güçlü sinyal | Bu repoda |
|--------------|---------------------------|-----------|
| **Architecture & Design** | Modüller bağımsız değiştirilebilir; tradeoff yazılı | `docs/ARCHITECTURE.md`, `src/algorithms/*` vs `src/flask_app/*`, `scripts/run_*.py` |
| **Implementation Quality** | Algoritma doğruluğu + sınır durumlar + debug karşılaştırma | KD-tree vs `knn_bruteforce`, `SpatiotemporalKNN(use_bruteforce=...)`, `tests/test_kdtree_knn.py`, mesafe/split testleri |
| **Development Process** | PR disiplini + anlamlı test + otomasyon | `docs/CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/workflows/ci.yml` |
| **Final Product** | Etkileşimli, tutarlı uçtan uca akış | Flask nav + `index.html` akışı; `/viz/hotspots`, `/viz/knn`, `/api/knn/predict`; dashboard route’ları |

**Not:** Nihai not rubric + retrospective + ekip sürecine bağlıdır; bu tablo “kanıt dosyalarını” toparlar.
