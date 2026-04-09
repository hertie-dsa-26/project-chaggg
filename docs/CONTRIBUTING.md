# Contributing — Development Process (rubric)

Bu depo için beklenen profesyonel iş akışı. **Exceeds** seviyesinde özellikle **PR incelemesi**, **anlamlı commit** ve **CI yeşili** somut kanıt oluşturur.

## Dal ve PR

1. `main`’e **doğrudan push yok**; özellik/issue başına branch (`feature/…`, `fix/…`).
2. PR açınca **en az bir reviewer** atayın; onay olmadan merge yok.
3. PR açıklaması: **ne**, **neden**, **nasıl test edildi** (komut veya ekran görüntüsü).
4. Büyük değişiklikleri küçük commit’lere bölün; mesajlar bağlam içersin (`fix: …`, `feat: …` tutarlı kullanımı tercih).

## Yerel kontroller (PR öncesi)

```bash
uv sync --all-groups
uv run ruff check src tests
uv run ruff format --check src tests
CHAGGG_SKIP_DATA_LOAD=1 uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Lint/format şu an **`src/` ve `tests/`** ile sınırlı; `scripts/` içindeki eski `from config import *` kalıpları ayrı bir temizlik PR’ında genişletilebilir.

KNN veya Flask KNN sayfası için (opsiyonel artefakt):

```bash
uv run python scripts/run_knn_prep.py --write-split
uv run python scripts/run_knn_forecast.py --k 5,10,20 --space meters --time-scale 0
```

## CI

GitHub Actions: `uv sync --locked` ve unit testler her push/PR’da çalışır. Kırmızı CI’da merge etmeyin.

## Kod incelemesinde bakılacaklar

- Yeni davranış için test veya gerekçeli istisna.
- Performans kritik yolda gereksiz kopya / tam veri yükleme yok mu.
- Flask: yeni route’da hızlı test (mümkünse `CHAGGG_SKIP_DATA_LOAD` uyumlu).

## Dokümantasyon

Mimari özeti: `docs/ARCHITECTURE.md`. Özellik bazlı yaşayan raporlar: `docs/US-*.md`.
