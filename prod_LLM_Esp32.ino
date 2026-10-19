// ---------------------------------------------------------------------------
//  Mini-LLM auf dem ESP32 - Chat ueber UART (Serial Monitor)
//
//  Board:          ESP32 Dev Module (oder kompatibel)
//  Serial Monitor: 115200 Baud, Zeilenende "Neue Zeile" (NL)
//
//  Das Modell liegt INT8-quantisiert komplett im Flash, die Inferenz laeuft
//  auf dem CPU-Kern - keine Cloud, kein WLAN. Der KV-Cache ist ebenfalls
//  INT8 und braucht damit nur ein Viertel des RAM einer float-Variante.
//
//  Befehle im Chat:
//      /reset          Gespraech vergessen
//      /temp 0.8       Temperatur setzen (0 = greedy, immer gleiche Antwort)
//      /topk 40        Top-k setzen (0 = aus)
//      /penalty 1.1    Wiederholungsstrafe (1.0 = aus)
//      /verlauf 1      Gespraechsverlauf mitschicken (0 = jede Frage einzeln)
//      /kontext        zeigt, was zuletzt im Kontextfenster stand
//      /info           Speicher- und Modellinfos
//      /help           Hilfe
// ---------------------------------------------------------------------------

// Alle Einstellungen - Architektur, KV-Cache-Quantisierung, Sampling,
// Puffergroessen - stehen in llm_config.h. Wer einen Wert nur fuer diesen
// Sketch aendern will, kann ihn auch hier VOR dem #include setzen, z.B.:
//     #define LLM_KV_MODE      LLM_KV_INT16
//     #define LLM_CONTEXT_LEN  32
#include "llm_config.h"
#include "llm_chat.h"

// Waehrend der Generierung regelmaessig dem Scheduler/Watchdog Luft geben
#if LLM_YIELD_DURING_GEN && defined(ESP32)
  #define LLM_YIELD() yield()
#else
  #define LLM_YIELD()
#endif

// --------------------------- Laufzeit-Parameter ----------------------------
// Startwerte aus llm_config.h, zur Laufzeit ueber /temp, /topk, /penalty
// und /verlauf umstellbar.
static float g_temperature = LLM_DEF_TEMPERATURE;
static int   g_top_k       = LLM_DEF_TOP_K;
static float g_repeat_pen  = LLM_DEF_REPEAT_PENALTY;
static int   g_use_history = LLM_DEF_USE_HISTORY;

// --------------------------- Gespraechsverlauf -----------------------------
static int  g_hist[LLM_HIST_MAX];
static int  g_hist_n = 0;
static int  g_win_start = 0;   // Beginn des zuletzt gesendeten Fensters in g_hist

static char g_input[LLM_INPUT_BUF_SIZE];
static int  g_input_n = 0;

static void hist_push(int tok)
{
    if (g_hist_n >= LLM_HIST_MAX) {
        // aeltere Haelfte verwerfen
        int keep = LLM_HIST_MAX / 2;
        memmove(g_hist, g_hist + (LLM_HIST_MAX - keep), sizeof(int) * keep);
        g_hist_n = keep;
    }
    g_hist[g_hist_n++] = tok;
}

// --------------------------- Antwort erzeugen ------------------------------
static void generate_reply(void)
{
    // Prompt = moeglichst langer Ausschnitt vom Ende des Verlaufs
    // llm_core.h stellt per #error sicher, dass hier mindestens 1 Token
    // fuer den Prompt uebrig bleibt.
    int max_prompt = LLM_CONTEXT_LEN - LLM_GEN_RESERVE;

    int start = g_hist_n - max_prompt;
    if (start < 0) start = 0;
    int n_prompt = g_hist_n - start;
    g_win_start = start;   // fuer /kontext merken

    // Prefill: den Prompt Token fuer Token durch das Modell schicken
    llm_reset_context();
    const float *logits = NULL;
    for (int i = 0; i < n_prompt; ++i) {
        logits = llm_forward(g_hist[start + i]);
        LLM_YIELD();
    }

    Serial.print(F("LLM > "));

    uint32_t t0 = millis();
    int generated = 0;
    int recent[LLM_REPEAT_WINDOW];
    int n_recent = 0;

    while (logits && generated < LLM_MAX_NEW_TOKENS) {
        int tok = llm_sample(logits, g_temperature, g_top_k,
                             recent, n_recent, g_repeat_pen);

        if (tok == TOK_END || tok == TOK_USER || tok == TOK_ASSISTANT) break;

        if (generated > 0 && !llm_no_space_before(tok)) Serial.print(' ');
        Serial.print(llm_decode(tok));

        hist_push(tok);
        if (n_recent < (int)(sizeof(recent) / sizeof(recent[0]))) {
            recent[n_recent++] = tok;
        } else {
            memmove(recent, recent + 1, sizeof(int) * (n_recent - 1));
            recent[n_recent - 1] = tok;
        }
        generated++;

        LLM_YIELD();
        logits = llm_forward(tok);   // NULL, wenn der Kontext voll ist
    }

    uint32_t dt = millis() - t0;
    hist_push(TOK_END);

    Serial.println();
#if LLM_SHOW_STATS
    if (generated > 0) {
        Serial.printf("      [%d Token in %lu ms = %.1f Token/s | Kontext %d/%d]\n",
                      generated, (unsigned long)dt,
                      dt ? 1000.0f * generated / dt : 0.0f,
                      llm_context_len(), LLM_CONTEXT_LEN);
    } else {
        Serial.println(F("      [keine Antwort - /reset probieren]"));
    }
#else
    (void)dt;
#endif
}

// --------------------------- Eingabe verarbeiten ---------------------------
static void print_help(void)
{
    Serial.println(F("Befehle:"));
    Serial.println(F("  /reset         Gespraech vergessen"));
    Serial.println(F("  /temp <wert>   Temperatur (0 = greedy, 0.7-1.0 kreativ)"));
    Serial.println(F("  /topk <n>      Top-k Sampling (0 = aus)"));
    Serial.println(F("  /penalty <w>   Wiederholungsstrafe (1.0 = aus)"));
    Serial.println(F("  /verlauf <0|1> Gespraechsverlauf mitschicken"));
    Serial.println(F("  /kontext       zeigt das Kontextfenster der letzten Antwort"));
    Serial.println(F("  /kontext ids   dasselbe mit Token-Nummern"));
    Serial.println(F("  /info          Modell- und Speicherinfos"));
    Serial.println(F("  /help          diese Hilfe"));
    Serial.println();
}

static void print_info(void)
{
    Serial.println(F("--- Modell ---"));
    Serial.printf("  Parameter     : %d (INT8)\n", LLM_PARAMETER_COUNT);
    Serial.printf("  Vokabular     : %d Token\n",  LLM_VOCAB_SIZE);
    Serial.printf("  Embedding/Dim : %d, Heads: %d, Layer: %d\n",
                  LLM_EMBEDDING_DIM, LLM_NUM_HEADS, LLM_NUM_LAYERS);
    Serial.printf("  Kontext       : %d von %d Token\n",
                  LLM_CONTEXT_LEN, LLM_BLOCK_SIZE);
    Serial.printf("  Architektur   : %s + %s\n",
                  LLM_PRE_NORM ? "Pre-Norm" : "Post-Norm", LLM_FF_ACT_NAME);
    Serial.println(F("--- Laufzeit ---"));
    Serial.printf("  Freier Heap   : %u Byte\n", (unsigned)ESP.getFreeHeap());
    Serial.printf("  KV-Cache      : %u Byte (%s, %d Skala/Pos)\n",
                  (unsigned)llm_kv_bytes(), LLM_KV_MODE_NAME, LLM_KV_GROUPS);
    Serial.printf("  Verlauf       : %d Token\n", g_hist_n);
    Serial.printf("  temp=%.2f  topk=%d  penalty=%.2f  verlauf=%d\n",
                  g_temperature, g_top_k, g_repeat_pen, g_use_history);
    Serial.println();
}

// Gibt das zuletzt an das Modell geschickte Kontextfenster aus - also genau
// die Tokenfolge, auf der die letzte Antwort beruht, inklusive Spezialtoken.
static void print_context(int with_ids)
{
    const int n    = g_hist_n - g_win_start;   // Token im Fenster
    int       free = LLM_CONTEXT_LEN - n;
    if (free < 0) free = 0;

    Serial.printf("--- Kontextfenster: %d von %d Token ---\n", n, LLM_CONTEXT_LEN);

    if (n <= 0) {
        Serial.println(F("  (leer - noch nichts gesendet)"));
        Serial.println();
        return;
    }

    if (g_win_start > 0) {
        Serial.printf("  ... %d aeltere Token abgeschnitten\n", g_win_start);
    }

    for (int i = g_win_start; i < g_hist_n; ++i) {
        const int tok = g_hist[i];

        // Sprecherwechsel bekommt eine eigene Zeile
        if (tok == TOK_USER || tok == TOK_ASSISTANT) {
            if (i > g_win_start) Serial.println();
            Serial.print(F("  "));
            Serial.print(llm_decode(tok));
        } else {
            if (!llm_no_space_before(tok)) Serial.print(' ');
            Serial.print(llm_decode(tok));
        }

        if (with_ids) Serial.printf("[%d]", tok);
    }

    Serial.println();
    Serial.printf("  (%d Token frei. Das letzte <END> zaehlt erst beim naechsten Durchlauf mit.)\n\n",
                  free);
}

// gibt 1 zurueck, wenn die Zeile ein Befehl war
static int handle_command(const char *line)
{
    if (line[0] != '/') return 0;

    if (strncmp(line, "/reset", 6) == 0) {
        g_hist_n = 0;
        llm_reset_context();
        Serial.println(F("[Gespraech zurueckgesetzt]\n"));
    } else if (strncmp(line, "/temp", 5) == 0) {
        g_temperature = (float)atof(line + 5);
        if (g_temperature < 0.0f) g_temperature = 0.0f;
        Serial.printf("[Temperatur = %.2f]\n\n", g_temperature);
    } else if (strncmp(line, "/topk", 5) == 0) {
        g_top_k = atoi(line + 5);
        if (g_top_k < 0) g_top_k = 0;
        Serial.printf("[Top-k = %d]\n\n", g_top_k);
    } else if (strncmp(line, "/penalty", 8) == 0) {
        g_repeat_pen = (float)atof(line + 8);
        if (g_repeat_pen < 1.0f) g_repeat_pen = 1.0f;
        Serial.printf("[Wiederholungsstrafe = %.2f]\n\n", g_repeat_pen);
    } else if (strncmp(line, "/verlauf", 8) == 0) {
        g_use_history = atoi(line + 8) ? 1 : 0;
        g_hist_n = 0;
        Serial.printf("[Verlauf %s]\n\n", g_use_history ? "an" : "aus");
    } else if (strncmp(line, "/kontext", 8) == 0) {
        // "/kontext ids" zeigt zusaetzlich die Token-Nummern
        print_context(strstr(line, "ids") != NULL);
    } else if (strncmp(line, "/info", 5) == 0) {
        print_info();
    } else {
        print_help();
    }
    return 1;
}

static void handle_line(char *line)
{
    // Leerzeichen am Ende entfernen
    int n = strlen(line);
    while (n > 0 && (line[n - 1] == ' ' || line[n - 1] == '\r')) line[--n] = '\0';
    if (n == 0) return;

    if (handle_command(line)) return;

    // Ohne Verlaufs-Modus startet jede Frage mit leerem Kontext
    if (!g_use_history) g_hist_n = 0;

    // <USER> ... <ASSISTANT> in den Verlauf schreiben
    int toks[LLM_CONTEXT_LEN];
    int nt = llm_encode(line, toks, LLM_CONTEXT_LEN);
    if (nt == 0) return;

#if LLM_UNK_REPORT_MAX > 0
    // Woerter, die das Modell nicht kennt, vor der Antwort melden. Sie gehen
    // trotzdem als <UNK> mit in den Prompt - die Antwort wird dadurch aber
    // ungenauer, weil dem Modell an der Stelle die Bedeutung fehlt.
    for (int i = 0; i < llm_unk_n; ++i) {
        Serial.printf("      [Wort nicht im Vokabular: \"%s\"]\n",
                      llm_unk_words[i]);
    }
    if (llm_unk_total > llm_unk_n) {
        Serial.printf("      [... und %d weitere nicht im Vokabular]\n",
                      llm_unk_total - llm_unk_n);
    }
#endif

    hist_push(TOK_USER);
    for (int i = 0; i < nt; ++i) hist_push(toks[i]);
    hist_push(TOK_ASSISTANT);

    generate_reply();
    Serial.println();
}

// --------------------------- Arduino ---------------------------------------
void setup()
{
    Serial.begin(LLM_SERIAL_BAUD);
    delay(300);

    llm_chat_init();

    Serial.println();
    Serial.println(F("========================================"));
    Serial.println(F("  Mini-LLM auf dem ESP32 - offline Chat"));
    Serial.println(F("========================================"));
    Serial.printf("  %d Parameter INT8, Kontext %d Token\n",
                  LLM_PARAMETER_COUNT, LLM_CONTEXT_LEN);
    Serial.printf("  Heap vor  Allokation: %u Byte\n", (unsigned)ESP.getFreeHeap());

    // KV-Cache auf dem Heap anlegen (Groesse siehe llm_config.h)
    if (!llm_init()) {
        Serial.printf("  FEHLER: %u Byte fuer den KV-Cache nicht verfuegbar!\n",
                      (unsigned)llm_kv_bytes());
        Serial.println(F("  Tipp: WLAN/Bluetooth aus lassen, in llm_config.h"));
        Serial.println(F("        LLM_CONTEXT_LEN senken oder LLM_USE_PSRAM aktivieren."));
        while (true) delay(1000);
    }
    Serial.printf("  KV-Cache: %u Byte (%s), Heap danach: %u Byte\n",
                  (unsigned)llm_kv_bytes(), LLM_KV_MODE_NAME,
                  (unsigned)ESP.getFreeHeap());

    Serial.println(F("  Einfach auf Deutsch schreiben, /help fuer Befehle."));
    Serial.println(F("  (Serial Monitor: 115200 Baud, Zeilenende = NL)"));
    Serial.println();

#if LLM_SELFTEST
    // Kurzer Selbsttest: ein Token durchs Modell schicken
    uint32_t t0 = micros();
    llm_reset_context();
    llm_forward(TOK_USER);
    Serial.printf("  Selbsttest ok - 1 Token in %lu us\n\n",
                  (unsigned long)(micros() - t0));
    llm_reset_context();
#endif

    Serial.print(F("Du  > "));
}

void loop()
{
    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c == '\n' || c == '\r') {
            if (g_input_n == 0) continue;       // leere Zeile / CRLF ignorieren
            g_input[g_input_n] = '\0';
            g_input_n = 0;
            Serial.println();
            handle_line(g_input);
            Serial.print(F("Du  > "));
        } else if (g_input_n < LLM_INPUT_BUF_SIZE - 1) {
            g_input[g_input_n++] = c;
#if LLM_ECHO_INPUT
            Serial.print(c);                    // lokales Echo
#endif
        }
    }
}
