# DBSCAN hotspot — iyileştirme workflow’u

Bu dosya US-2.6 (DBSCAN) için **sırayla** ilerleyebileceğin aşamaları tanımlar. Her fazın sonunda “Done” kutularını işaretle; takım review’unda da aynı maddeyi PR açıklamasına kısaca yaz.

---

## Faz 0 — Baseline (mevcut durum)

**Amaç:** Şu anki pipeline’ı tekrar üretilebilir şekilde sabitle.

| # | Görev | Done |
|---|--------|------|
| 0.1 | `gasyar/us-2.6-dbscan` (veya eşdeğeri) branch’te çalış | ☐ |
| 0.2 | Temiz veri + `uv run python scripts/run_dbscan_hotspots.py` ile `outputs/dbscan/*` üret | ☐ |
| 0.3 | `metrics.json` içindeki `run_config` + `grid_search` tablosunu sakla (rapor/ek) | ☐ |

**Komut (referans):**

```bash
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplot uv run python scripts/run_dbscan_hotspots.py
```

---

## Faz 1 — Dokümantasyon ve şeffaflık (düşük efor, yüksek fayda)

**Amaç:** Hoca/reviewer’a “nasıl ölçtük?” sorusuna tek yerden cevap.

| # | Görev | Done |
|---|--------|------|
| 1.1 | PR veya wiki’de kısa blok: yüklü sütunlar, train yılları, negatif stratejisi (`sparse_grid` / `distance`) | ☐ |
| 1.2 | Negatif örnekleme: ızgara boyutu (`32×32`) ve `max_crimes_per_cell=0` anlamını 2 cümleyle açıkla | ☐ |
| 1.3 | Train alt örneği: varsayılan `max-train=45000` — final rapor için bilinçli olarak `--max-train 120000` koştuysan ikisini de kaydet | ☐ |

**İsteğe bağlı komutlar:**

```bash
# Karşılaştırma: eski negatif mantığı
uv run python scripts/run_dbscan_hotspots.py --negative-strategy distance --output-dir outputs/dbscan_distance_baseline
```

---

## Faz 2 — Mekânsal sıkılık (orta efor)

**Amaç:** Derece yerine veya dereceyle birlikte **metre** uzayında DBSCAN dene; raporda “limitasyon + iyileştirme” bölümü.

| # | Görev | Done |
|---|--------|------|
| 2.1 | Chicago için uygun projeksiyon seç (ör. UTM 16N `EPSG:32616` veya eyalet uyumlu bir CRS — takım CA/geo ile hizala) | ☐ |
| 2.2 | `lat/lon` → metre `xy` dönüşümü (geopandas / pyproj); DBSCAN’i **metre** `eps` ile çalıştıran bir mod veya script bayrağı ekle | ☐ |
| 2.3 | Story’de istenen **derece** grid’i (0.01, 0.05, 0.1) koru; metre denemelerini **ek tablo** olarak `metrics` altına yaz | ☐ |
| 2.4 | Küme sınırını hâlâ WGS84 GeoJSON’a çevirerek kaydet (web haritası uyumu) | ☐ |

**Done kriteri:** Aynı veride en az bir “degree-only” ve bir “projected meters” koşusu; kısa karşılaştırma paragrafı.

---

## Faz 3 — Değerlendirme çeşitliliği (orta efor)

**Amaç:** Binary metriği tek tanıma kilitleme.

| # | Görev | Done |
|---|--------|------|
| 3.1 | `max_crimes_per_cell`: 0 ile 1 (veya 2) dene; skor ve yorum farkını not et | ☐ |
| 3.2 | Izgara çözünürlüğü: `24×24` vs `32×32` — duyarlılık kontrolü | ☐ |
| 3.3 | İsteğe bağlı: precision–recall özeti veya eşik yok (F1 yeterliyse atla) | ☐ |

**Komut örnekleri:**

```bash
uv run python scripts/run_dbscan_hotspots.py --max-crimes-per-cell 1 --output-dir outputs/dbscan_cell1
uv run python scripts/run_dbscan_hotspots.py --grid-lat-bins 24 --grid-lon-bins 24 --output-dir outputs/dbscan_grid24
```

---

## Faz 4 — Ürün entegrasyonu (takım ihtiyacına göre)

**Amaç:** Flask veya sunumda kullanılabilir çıktı.

| # | Görev | Done |
|---|--------|------|
| 4.1 | `cluster_boundaries.geojson`’u uygulama veya statik haritada aç (Folium / kepler / frontend) | ☐ |
| 4.2 | Notebook’ta `contextily` veya eşdeğeri ile basemap + boundaries PNG | ☐ |
| 4.3 | İsteğe bağlı: `community_area` sütunu ile küme/CA özet CSV (script’e `usecols` genişletmesi gerekir) | ☐ |

---

## Faz 5 — İleri (opsiyonel, yüksek efor)

**Amaç:** Araştırma / bonus kalitesi; issue kapanması için şart değil.

| # | Görev | Done |
|---|--------|------|
| 5.1 | Concave hull / alpha shape ile sınır | ☐ |
| 5.2 | `primary_type` ile stratified veya ayrı DBSCAN koşuları | ☐ |
| 5.3 | Zaman dilimli (aylık) hotspot kararlılığı | ☐ |

---

## Özet sıra (öneri)

1. **Faz 0** → PR merge hazırlığı  
2. **Faz 1** → review / rapor savunması  
3. **Faz 2 veya 3** → birini seç (mekân doğruluğu mu, metrik sağlamlığı mı?)  
4. **Faz 4** → ekip “harita göster” diyorsa  
5. **Faz Sadece zaman kalırsa**

---

## PR / issue yorumu için şablon

```
Workflow: docs/dbscan-improvement-workflow.md
Tamamlanan fazlar: F0 ☐ F1 ☐ F2 ☐ …
Bu PR kapsamı: [kısa madde listesi]
Bilinen limitasyon: [ör. degree EPS, train subsample]
```

Bu metni PR açıklamasına yapıştırıp faz kutularını güncellemek yeterli.
