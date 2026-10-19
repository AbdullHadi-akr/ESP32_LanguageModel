// ---------------------------------------------------------------------------
//  llm_config.h - ALLE Einstellungen an einer Stelle
//
//  Hier stehen saemtliche Stellschrauben von Modell, KV-Cache, Sampling und
//  Chat-Oberflaeche. Jede Option laesst sich auf drei Wegen setzen:
//
//    1. direkt in dieser Datei aendern,
//    2. VOR dem #include "llm_chat.h" im Sketch definieren
//       (jeder Wert steht in einem #ifndef, der Sketch gewinnt also), oder
//    3. als Compiler-Flag  -DLLM_KV_MODE=LLM_KV_INT16
//
//  ACHTUNG bei den Architektur-Schaltern: die muessen zu den Gewichten in
//  model_int8.h passen. Falsch eingestellt laeuft alles durch, das Modell
//  redet aber nur noch Unsinn. Die aktuellen Werte sind gegen die Gewichte
//  getestet (siehe Kommentar bei LLM_PRE_NORM).
//
//  Die festen Groessen des Modells - LLM_VOCAB_SIZE, LLM_EMBEDDING_DIM,
//  LLM_NUM_HEADS, LLM_NUM_LAYERS, LLM_BLOCK_SIZE - stehen NICHT hier,
//  sondern in model_int8.h. Sie stecken in den Gewichten und lassen sich
//  nicht umkonfigurieren, nur neu trainieren.
// ---------------------------------------------------------------------------
#pragma once

// ===========================================================================
//  1. ARCHITEKTUR  (muss zu den Gewichten passen)
// ===========================================================================

// Reihenfolge von LayerNorm und Attention/FeedForward.
//   1 = Pre-Norm  (nanoGPT-Stil):  x = x + attn(norm1(x))
//   0 = Post-Norm (Original 2017): x = norm1(x + attn(x))
//
// Fuer die beiliegenden Gewichte ist 1 richtig. Per Host-Test verifiziert:
// mit Post-Norm faellt die mittlere Log-Wahrscheinlichkeit von -6.2 auf -9.0
// und das Modell gibt nur noch <ASSISTANT> aus.
#ifndef LLM_PRE_NORM
#define LLM_PRE_NORM 1
#endif

// Aktivierung im FeedForward. Mit diesen Gewichten liefern GELU und ReLU
// dieselben Antworten (Unterschied 0.004 nats ueber 163 Token), die Wahl
// laesst sich aus den Gewichten also nicht ablesen. GELU ist der uebliche
// Default fuer diese Modellgroesse.
#define LLM_ACT_GELU 0   // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
#define LLM_ACT_RELU 1   // max(0, x)            - am schnellsten
#define LLM_ACT_SILU 2   // x*sigmoid(x), "Swish"

#ifndef LLM_FF_ACT
#define LLM_FF_ACT LLM_ACT_GELU
#endif

// Breite der FeedForward-Zwischenschicht. Steckt in den Gewichten
// (blocks_N_ff_fc_in_weight_rows) - nur aendern, wenn das Modell neu
// exportiert wurde. llm_core.h prueft den Wert gegen model_int8.h.
#ifndef LLM_FF_DIM
#define LLM_FF_DIM 512
#endif

// Epsilon im LayerNorm. torch.nn.LayerNorm nutzt 1e-5.
#ifndef LLM_LN_EPS
#define LLM_LN_EPS 1e-5f
#endif

// ===========================================================================
//  2. KV-CACHE  (Genauigkeit gegen RAM)
// ===========================================================================
//
// Der Cache haelt Key und Value jeder bereits gesehenen Position, damit pro
// neuem Token nicht der ganze Kontext neu gerechnet werden muss. Er ist der
// mit Abstand groesste Posten im RAM und waechst linear mit dem Kontext.
//
// Groesse = 2 (K+V) * Layer * Kontext * Embedding * sizeof(Elementtyp)
//           + Skalen
//
// Bei 2 Layern, 64 Token Kontext und 128 Dimensionen:
//
//   Modus        Werte     Skalen   gesamt   Qualitaet
//   ---------------------------------------------------------------
//   INT8          32 KB      4 KB    36 KB   Standard, Verlust winzig
//   INT16         64 KB      4 KB    68 KB   praktisch wie float
//   FLOAT        128 KB      0 KB   128 KB   Referenz, nur zum Vergleichen
//
// Auf einem ESP32 ohne PSRAM ist INT8 die richtige Wahl - FLOAT passt bei
// groesserem Kontext schlicht nicht mehr in den Heap.

#define LLM_KV_INT8  0   // 1 Byte je Wert, Skala je Gruppe und Position
#define LLM_KV_INT16 1   // 2 Byte je Wert, Skala je Gruppe und Position
#define LLM_KV_FLOAT 2   // 4 Byte je Wert, keine Quantisierung

#ifndef LLM_KV_MODE
#define LLM_KV_MODE LLM_KV_INT8
#endif

// Feinheit der Quantisierung (wirkt nur bei INT8/INT16).
//   1 = eine Skala pro Kopf und Position  (4 Skalen je Position)
//       Genauer, weil ein Kopf mit grossen Werten die anderen nicht mit
//       herunterzieht. Kostet hier 4 KB.
//   0 = eine Skala pro Position           (1 Skala je Position)
//       Spart 3 KB, dafuer mehr Quantisierungsrauschen.
#ifndef LLM_KV_SCALE_PER_HEAD
#define LLM_KV_SCALE_PER_HEAD 1
#endif

// Nutzbare Kontextlaenge in Token. Darf hoechstens LLM_BLOCK_SIZE sein -
// so viele Positions-Embeddings hat das Modell gelernt. Kleiner stellen
// spart KV-Cache (linear!), das Modell vergisst dann aber frueher.
//   LLM_BLOCK_SIZE (64) -> 36 KB KV-Cache
//   32                  -> 18 KB
#ifndef LLM_CONTEXT_LEN
#define LLM_CONTEXT_LEN LLM_BLOCK_SIZE
#endif

// ===========================================================================
//  3. SPEICHER
// ===========================================================================

// Externes PSRAM als Rueckfallebene benutzen, wenn der interne Heap nicht
// reicht. Braucht ein Board mit PSRAM und im Boardmenue aktiviertes PSRAM.
// Standard: automatisch an, wenn der Core BOARD_HAS_PSRAM meldet.
#ifndef LLM_USE_PSRAM
  #ifdef BOARD_HAS_PSRAM
    #define LLM_USE_PSRAM 1
  #else
    #define LLM_USE_PSRAM 0
  #endif
#endif

// 1 = KV-Cache bevorzugt ins PSRAM legen und den knappen internen DRAM
//     fuer alles andere frei halten. PSRAM ist langsamer, bei diesem
//     Modell kostet das spuerbar Token/s.
// 0 = erst internen Heap versuchen, PSRAM nur als Notnagel (empfohlen).
#ifndef LLM_KV_PSRAM_FIRST
#define LLM_KV_PSRAM_FIRST 0
#endif

// Top-k-Sampling einkompilieren. Braucht einen zweiten Puffer von
// LLM_VOCAB_SIZE floats (bei 1325 Token = 5.3 KB statisches RAM).
// Auf 0 setzen spart das RAM, dann wirkt der Parameter /topk nicht mehr.
#ifndef LLM_ENABLE_TOP_K
#define LLM_ENABLE_TOP_K 1
#endif

// ===========================================================================
//  4. SAMPLING - Startwerte
// ===========================================================================
// Alle vier lassen sich zur Laufzeit ueber /temp, /topk, /penalty und
// /verlauf umstellen; hier steht nur, womit der Sketch startet.

// 0.0 = greedy (immer dieselbe Antwort), 0.7-1.0 = kreativ, >1.2 = wirr
#ifndef LLM_DEF_TEMPERATURE
#define LLM_DEF_TEMPERATURE 0.75f
#endif

// Nur aus den k wahrscheinlichsten Token ziehen. 0 = aus.
#ifndef LLM_DEF_TOP_K
#define LLM_DEF_TOP_K 10
#endif

// Daempft Token, die gerade schon kamen. 1.0 = aus, 1.1-1.3 sinnvoll.
#ifndef LLM_DEF_REPEAT_PENALTY
#define LLM_DEF_REPEAT_PENALTY 1.10f
#endif

// Wie viele der zuletzt erzeugten Token die Wiederholungsstrafe anschaut.
#ifndef LLM_REPEAT_WINDOW
#define LLM_REPEAT_WINDOW 16
#endif

// 0 = jede Frage wird einzeln beantwortet (empfohlen - das Modell wurde auf
//     einzelne Frage/Antwort-Paare trainiert und wird mit mitlaufendem
//     Verlauf schlechter)
// 1 = bisheriger Gespraechsverlauf wird mitgeschickt
#ifndef LLM_DEF_USE_HISTORY
#define LLM_DEF_USE_HISTORY 0
#endif

// ===========================================================================
//  5. ANTWORTLAENGE
// ===========================================================================

// Token, die im Kontextfenster fuer die Antwort frei bleiben. Der Rest
// steht dem Prompt zur Verfuegung:  Prompt = LLM_CONTEXT_LEN - Reserve.
//
// WICHTIG: muss kleiner als LLM_CONTEXT_LEN sein. War der Wert groesser,
// blieb genau 1 Token Prompt uebrig und das Modell hat die Frage gar nicht
// gesehen - llm_core.h faengt das jetzt mit einem #error ab.
#ifndef LLM_GEN_RESERVE
#define LLM_GEN_RESERVE 24
#endif

// Harte Obergrenze pro Antwort. Die Generierung endet ohnehin, sobald
// <END> kommt oder das Kontextfenster voll ist.
#ifndef LLM_MAX_NEW_TOKENS
#define LLM_MAX_NEW_TOKENS 64
#endif

// ===========================================================================
//  6. CHAT UND SERIELLE SCHNITTSTELLE
// ===========================================================================

#ifndef LLM_SERIAL_BAUD
#define LLM_SERIAL_BAUD 115200
#endif

// Token-Speicher des Gespraechs (nur bei /verlauf 1 wirklich genutzt).
// Kostet LLM_HIST_MAX * 4 Byte.
#ifndef LLM_HIST_MAX
#define LLM_HIST_MAX 768
#endif

// Groesster Eingabesatz in Zeichen.
#ifndef LLM_INPUT_BUF_SIZE
#define LLM_INPUT_BUF_SIZE 256
#endif

// Groesstes einzelnes Wort in Zeichen (Tokenizer-Puffer).
#ifndef LLM_WORD_BUF_SIZE
#define LLM_WORD_BUF_SIZE 64
#endif

// Unbekannte Woerter melden. Steht ein eingetipptes Wort nicht im Vokabular,
// gibt der Sketch "[Wort nicht im Vokabular: ...]" aus, bevor er antwortet.
// Der Wert ist die Zahl der Woerter, die pro Eingabe namentlich genannt
// werden - alles darueber wird nur noch gezaehlt. 0 = Meldung abschalten.
// Kostet LLM_UNK_REPORT_MAX * LLM_WORD_BUF_SIZE Byte statisches RAM
// (Standard: 8 * 64 = 512 Byte).
#ifndef LLM_UNK_REPORT_MAX
#define LLM_UNK_REPORT_MAX 8
#endif

// 1 = eingetippte Zeichen zurueckschicken. Auf 0 stellen, wenn das
//     Terminal selbst schon echoed und alles doppelt erscheint.
#ifndef LLM_ECHO_INPUT
#define LLM_ECHO_INPUT 1
#endif

// 1 = nach jeder Antwort die Zeile [N Token in X ms = Y Token/s] ausgeben
#ifndef LLM_SHOW_STATS
#define LLM_SHOW_STATS 1
#endif

// 1 = beim Start ein Token durchs Modell schicken und die Zeit messen
#ifndef LLM_SELFTEST
#define LLM_SELFTEST 1
#endif

// 1 = waehrend der Generierung yield() rufen, damit Watchdog und
//     WLAN-Stack Luft bekommen. Auf 0 nur, wenn sonst nichts laeuft.
#ifndef LLM_YIELD_DURING_GEN
#define LLM_YIELD_DURING_GEN 1
#endif
