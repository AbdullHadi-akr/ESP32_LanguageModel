# Mini-LLM auf dem ESP32

Ein kleines Transformer-Sprachmodell (GPT-Stil), das **komplett offline auf einem ESP32** läuft.
Gechattet wird über den Serial Monitor – keine Cloud, kein WLAN, keine Frameworks.

*A tiny GPT-style transformer running fully offline on an ESP32, written in plain C/C++. Chat with it in German over the serial port.*

```
Du  > Hallo, wie geht es dir?
LLM > ...
      [N Token in X ms = Y Token/s | Kontext n/64]
```

## Eckdaten

| | |
|---|---|
| Architektur | Decoder-only Transformer, Pre-Norm, GELU, Weight Tying |
| Größe | 2 Layer · 4 Heads · 128 Dimensionen · FeedForward 512 |
| Parameter | ca. 575 000 |
| Gewichte | INT8, eine float-Skala pro Zeile, liegen im Flash |
| Kontext | 64 Token |
| Vokabular | 1 325 Token, wortbasiert, Deutsch |
| KV-Cache | INT8 mit Skala pro Head und Position: 36 KB (statt 128 KB als float) |

## Features

- Inferenz in reinem C/C++ ohne externe Libraries (läuft auch auf dem PC)
- Quantisierter KV-Cache wahlweise in INT8, INT16 oder float
- Sampling mit Temperatur, Top-k und Repetition Penalty, zur Laufzeit einstellbar
- Optionale PSRAM-Nutzung für Boards mit externem RAM
- Alle Einstellungen zentral in [`llm_config.h`](llm_config.h)
- Prüfung der Konfiguration beim Kompilieren (`#error` statt Kauderwelsch)
- Meldung, wenn ein eingegebenes Wort nicht im Vokabular steht

## Loslegen

1. **Arduino IDE** mit dem [ESP32-Core von Espressif](https://github.com/espressif/arduino-esp32) installieren.
2. Den Ordner `prod_LLM_Esp32` öffnen (der Ordnername muss zum Sketch `prod_LLM_Esp32.ino` passen).
3. Board **ESP32 Dev Module** (oder kompatibel) auswählen.
   Bei großem Sketch ggf. ein Partitionsschema mit mehr App-Speicher wählen, z. B. *Huge APP*.
4. Hochladen, Serial Monitor öffnen: **115200 Baud**, Zeilenende **„Neue Zeile“**.
5. Auf Deutsch schreiben – `/help` zeigt alle Befehle.

## Befehle im Chat

| Befehl | Wirkung |
|---|---|
| `/reset` | Gespräch vergessen |
| `/temp 0.8` | Temperatur (0 = greedy, 0.7–1.0 kreativ) |
| `/topk 10` | Top-k Sampling (0 = aus) |
| `/penalty 1.1` | Wiederholungsstrafe (1.0 = aus) |
| `/verlauf 0\|1` | Gesprächsverlauf mitschicken |
| `/kontext` | zeigt das Kontextfenster der letzten Antwort (`/kontext ids` mit Token-Nummern) |
| `/info` | Modell- und Speicherinfos |
| `/help` | Hilfe |

## Konfiguration

Alle Stellschrauben stehen in [`llm_config.h`](llm_config.h). Jeder Wert kann dort geändert,
im Sketch **vor** dem `#include` überschrieben oder als Compiler-Flag gesetzt werden:

```cpp
#define LLM_KV_MODE      LLM_KV_INT16   // genauer, doppelter RAM-Bedarf
#define LLM_CONTEXT_LEN  32             // halbiert den KV-Cache
#include "llm_config.h"
#include "llm_chat.h"
```

> **Achtung:** Die Architektur-Schalter (`LLM_PRE_NORM`, `LLM_FF_ACT`, `LLM_FF_DIM`) müssen zu den
> Gewichten passen. Die festen Modellgrößen stehen in `model_int8.h` und lassen sich nur durch
> Neutrainieren ändern.

## Dateien

| Datei | Inhalt |
|---|---|
| `prod_LLM_Esp32.ino` | Sketch: Chat-Schleife, Befehle, Ausgabe |
| `llm_config.h` | alle Einstellungen |
| `llm_core.h` | Transformer-Inferenz, KV-Cache, Sampling |
| `llm_chat.h` | Tokenizer (Encode/Decode), Dialogformat `<USER> … <ASSISTANT> … <END>` |
| `tokenizer.h` | Vokabular (generiert) |
| `model_int8.h` | quantisierte Gewichte (generiert) |

## Grenzen

Das ist ein Lern- und Demoprojekt, kein ChatGPT: Das Modell ist winzig, kennt nur ~1 300 Wörter
und wurde auf einzelne Frage/Antwort-Paare trainiert. Unbekannte Wörter werden zu `<UNK>`.

## Lizenz

[MIT](LICENSE)
