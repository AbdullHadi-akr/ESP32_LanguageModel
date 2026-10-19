// ---------------------------------------------------------------------------
//  llm_chat.h - Wort-Tokenizer (Encode/Decode) + Kontextverwaltung
//
//  Der Tokenizer ist wortbasiert: Text wird an Leerzeichen getrennt,
//  Satzzeichen (. , ! ? -) werden als eigene Token abgetrennt.
//  Dialogformat:  <USER> ... <ASSISTANT> ... <END>
// ---------------------------------------------------------------------------
#pragma once

#include "llm_core.h"

// Ids der Spezialtoken. Die Startwerte kommen aus tokenizer.h, llm_chat_init()
// schlaegt sie zur Sicherheit noch einmal im Vokabular nach.
static int TOK_UNK       = LLM_UNK_ID,
           TOK_PAD       = LLM_PAD_ID,
           TOK_ASSISTANT = LLM_ASSISTANT_ID,
           TOK_END       = LLM_END_ID,
           TOK_USER      = LLM_USER_ID;

static void llm_chat_init(void)
{
    TOK_UNK       = llm_token_to_id("<UNK>");
    TOK_PAD       = llm_token_to_id("<PAD>");
    TOK_ASSISTANT = llm_token_to_id("<ASSISTANT>");
    TOK_END       = llm_token_to_id("<END>");
    TOK_USER      = llm_token_to_id("<USER>");
}

static inline int llm_is_punct(char c)
{
    return c == '.' || c == ',' || c == '!' || c == '?' || c == '-';
}

// --------------------------- Unbekannte Woerter ----------------------------
// llm_encode() vermerkt hier jedes Wort, das nicht im Vokabular steht, damit
// der Sketch es hinterher melden kann. Die Liste gilt immer nur fuer den
// letzten llm_encode()-Aufruf und wird dort zu Beginn geleert.
#if LLM_UNK_REPORT_MAX > 0

static char llm_unk_words[LLM_UNK_REPORT_MAX][LLM_WORD_BUF_SIZE];
static int  llm_unk_n     = 0;   // namentlich gemerkte Woerter
static int  llm_unk_total = 0;   // insgesamt gefundene (auch ueber dem Limit)

static void llm_unk_reset(void)
{
    llm_unk_n     = 0;
    llm_unk_total = 0;
}

// Merkt ein unbekanntes Wort vor. Wiederholungen innerhalb einer Eingabe
// werden nur einmal gezaehlt, damit die Meldung kurz bleibt.
static void llm_unk_record(const char *w)
{
    for (int i = 0; i < llm_unk_n; ++i) {
        if (strcmp(llm_unk_words[i], w) == 0) return;   // schon vermerkt
    }

    llm_unk_total++;
    if (llm_unk_n >= LLM_UNK_REPORT_MAX) return;        // nur noch zaehlen

    size_t n = strlen(w);
    if (n >= LLM_WORD_BUF_SIZE) n = LLM_WORD_BUF_SIZE - 1;
    memcpy(llm_unk_words[llm_unk_n], w, n);
    llm_unk_words[llm_unk_n][n] = '\0';
    llm_unk_n++;
}

#endif  // LLM_UNK_REPORT_MAX > 0

// Ein Wort nachschlagen: exakt -> klein geschrieben -> gross geschrieben
static int llm_lookup_word(const char *w)
{
    if (w[0] == '\0') return -1;

    int id = llm_token_to_id(w);
    if (id != TOK_UNK) return id;

    char buf[LLM_WORD_BUF_SIZE];
    size_t n = strlen(w);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, w, n);
    buf[n] = '\0';

    if (buf[0] >= 'A' && buf[0] <= 'Z') {
        buf[0] = (char)(buf[0] - 'A' + 'a');
        id = llm_token_to_id(buf);
        if (id != TOK_UNK) return id;
    } else if (buf[0] >= 'a' && buf[0] <= 'z') {
        buf[0] = (char)(buf[0] - 'a' + 'A');
        id = llm_token_to_id(buf);
        if (id != TOK_UNK) return id;
    }

    // Auch in der anderen Schreibweise nicht gefunden - in der
    // Originalschreibweise vermerken, wie der Nutzer es getippt hat.
#if LLM_UNK_REPORT_MAX > 0
    llm_unk_record(w);
#endif
    return TOK_UNK;
}

// Text -> Token-Ids. Rueckgabe: Anzahl der Token.
static int llm_encode(const char *text, int *out, int max_out)
{
    int n = 0;
    const char *p = text;
    char word[LLM_WORD_BUF_SIZE];

#if LLM_UNK_REPORT_MAX > 0
    llm_unk_reset();   // Liste gilt nur fuer diesen Aufruf
#endif

    while (*p && n < max_out) {
        // Leerzeichen ueberspringen
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p) break;

        // fuehrende Satzzeichen als eigene Token
        while (llm_is_punct(*p) && n < max_out) {
            char s[2] = { *p, '\0' };
            int id = llm_token_to_id(s);
            if (id != TOK_UNK) out[n++] = id;
            ++p;
        }

        // Wort einsammeln
        size_t w = 0;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n'
               && !llm_is_punct(*p) && w < sizeof(word) - 1) {
            word[w++] = *p++;
        }
        word[w] = '\0';

        if (w > 0 && n < max_out) {
            int id = llm_lookup_word(word);
            if (id >= 0) out[n++] = id;
        }

        // nachfolgende Satzzeichen als eigene Token
        while (llm_is_punct(*p) && n < max_out) {
            char s[2] = { *p, '\0' };
            int id = llm_token_to_id(s);
            if (id != TOK_UNK) out[n++] = id;
            ++p;
        }
    }
    return n;
}

static const char *llm_decode(int id)
{
    if (id < 0 || id >= LLM_VOCAB_SIZE) return "<UNK>";
    return llm_itos[id];
}

// true, wenn vor diesem Token KEIN Leerzeichen stehen soll
static int llm_no_space_before(int id)
{
    const char *s = llm_decode(id);
    return s[1] == '\0' && llm_is_punct(s[0]);
}

static int llm_is_special(int id)
{
    return id == TOK_UNK || id == TOK_PAD || id == TOK_END
        || id == TOK_USER || id == TOK_ASSISTANT;
}
