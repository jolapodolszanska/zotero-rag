# zotero-rag — Twoja biblioteka Zotero jako narzędzie dla Claude (MCP)

Własny, odchudzony pipeline RAG inspirowany
[RAG-Assistant-for-Zotero](https://github.com/Quiet-Signals-Lab/RAG-Assistant-for-Zotero)
(kod napisany od nowa). Serwer MCP przeszukuje pełne teksty PDF z Zotero,
a odpowiedź z cytowaniami pisze Claude (Desktop lub Code). Nie trzeba klucza API.
Wszystko liczy się lokalnie.

## Pipeline

```
zotero.sqlite (kopia, tylko odczyt) ──► PDF (PyMuPDF)
   ──► czyszczenie: nagłówki/stopki, dzielenie wyrazów, ligatury, bibliografia pomijana
   ──► fragmenty ~1000 znaków ze zdań w obrębie sekcji (+ abstrakt z Zotero)
   ──► embeddingi bge-m3 (wielojęzyczne: PL+EN, zapytanie PL znajduje tekst EN)
   ──► data/index.sqlite: wektory + FTS5/BM25 (bez polskich znaków diakrytycznych też działa)

zapytanie ──► wektory (cosine) + BM25 (z dopasowaniem prefiksów pod polską fleksję)
          ──► RRF ──► reranker bge-reranker-v2-m3 ──► max N fragmentów na pracę
          ──► Claude: odpowiedź z cytatami (Nazwisko, rok, s. X)
```

Pliki: `zotero_rag/zotero.py` (Zotero), `pdftext.py` (PDF→fragmenty), `store.py` (indeks),
`models.py` (modele), `search.py` (wyszukiwanie), `indexer.py`, `server.py` (narzędzia MCP), `cli.py`.

## Użycie

```powershell
git clone <adres-repo> zotero-rag
cd zotero-rag
uv sync                              # instalacja zależności
uv run zotero-rag index              # indeksuj nowe/zmienione PDF-y (--full: od zera)
uv run zotero-rag search "redukcja artefaktów w CBCT"   # test z terminala
```

### Podłączenie do Claude Code

```powershell
claude mcp add zotero --scope user "C:\ścieżka\do\zotero-rag\.venv\Scripts\zotero-rag.exe" serve
```

(macOS/Linux: `.venv/bin/zotero-rag`.)

(W PowerShellu nie używaj wariantu z `-- uv --directory ...`: PowerShell zjada `--` i `claude`
zgłasza `unknown option '--directory'`.)

### Podłączenie do Claude Desktop

`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "zotero": {
      "command": "uv",
      "args": ["--directory", "C:\\ścieżka\\do\\zotero-rag", "run", "zotero-rag", "serve"]
    }
  }
}
```

Po restarcie Claude zapytaj np. *„Co moja biblioteka mówi o wyjaśnialności (XAI) w diagnostyce Alzheimera? Cytuj strony.”*

## Narzędzia MCP

| narzędzie | do czego |
|---|---|
| `search_library` | wyszukiwanie hybrydowe; filtry: lata, autor, tag, kolekcja, typ |
| `search_within_item` | szukanie w jednej pracy |
| `get_context` | sąsiednie fragmenty (gdy cytat jest urwany) |
| `get_item` | pełny opis bibliograficzny + abstrakt |
| `list_items` | lista prac (z filtrami) |
| `index_status` | statystyki, kolekcje, tagi |
| `update_index` | doindeksowanie nowych PDF-ów |

## Ustawienia (zmienne środowiskowe)

| zmienna | domyślnie |
|---|---|
| `ZOTERO_DIR` | `~/Zotero` |
| `ZOTERO_BASE_DIR` | — (folder bazowy dla plików „linked”) |
| `ZOTERO_RAG_DATA` | `./data` |
| `EMBED_MODEL` | `BAAI/bge-m3` (zmiana = pełna reindeksacja) |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` (`""` wyłącza) |
| `EMBED_QUERY_PREFIX` / `EMBED_DOC_PREFIX` | — (np. `query: ` / `passage: ` dla multilingual-e5) |

Inne modele dobre dla polskiego: `sdadas/mmlw-roberta-large` (prefiks zapytania `zapytanie: `),
`intfloat/multilingual-e5-large` (prefiksy `query: ` / `passage: `). Lżejszy i szybszy na CPU:
`intfloat/multilingual-e5-base`.

PDF-y muszą mieć warstwę tekstową (skany bez OCR zostaną pominięte i zgłoszone w logu).

## Licencja

MIT, zob. [LICENSE](LICENSE). Pomysł zaczerpnięty z
[RAG-Assistant-for-Zotero](https://github.com/Quiet-Signals-Lab/RAG-Assistant-for-Zotero) (GPL-3.0);
kod tego repozytorium został napisany od nowa.
