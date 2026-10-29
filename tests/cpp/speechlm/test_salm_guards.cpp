// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Decoding guards (salm_guards): unit vectors for the loop, script and language guards.
#include <cstdio>
#include <string>
#include <vector>

#include "salm_guards.h"

using namespace nemo_speech::speechlm::guards;

static int failures = 0;
static void
check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

static std::vector<int>
cycle(std::vector<int> unit, size_t n) {
    std::vector<int> out;
    for (size_t i = 0; i < n; ++i) out.push_back(unit[i % unit.size()]);
    return out;
}

static int
period(const std::vector<int>& t) {
    return loop_period(t, t.size());
}

int
main() {
    // loop period vectors
    check(
        period(cycle({7}, 15)) == 0 && period(cycle({7}, 31)) == 0 && period(cycle({7}, 32)) == 1,
        "period 1");
    check(period(cycle({1, 2}, 32)) == 2 && period(cycle({1, 2}, 31)) == 0, "period 2");
    check(period(cycle({1, 2, 3}, 32)) == 3 && period(cycle({1, 2, 3}, 31)) == 0, "period 3");
    check(period(cycle({1, 2, 3, 4}, 16)) == 4 && period(cycle({1, 2, 3, 4}, 15)) == 0, "period 4");
    check(
        period(cycle({1, 2, 3, 4, 5}, 20)) == 5 && period(cycle({1, 2, 3, 4, 5}, 19)) == 0,
        "period 5");
    check(
        period(cycle({1, 2, 3, 4, 5, 6}, 24)) == 6 && period(cycle({1, 2, 3, 4, 5, 6}, 23)) == 0,
        "period 6");
    std::vector<int> u13;
    for (int i = 0; i < 13; ++i) u13.push_back(100 + i);
    check(period(cycle(u13, 52)) == 13 && period(cycle(u13, 51)) == 0, "period 13");

    std::vector<int> quote = {501, 502, 503, 504, 505, 506};  // a six-token phrase, three times
    check(
        period(cycle(quote, 18)) == 0 && period(cycle(quote, 24)) == 6,
        "a phrase said three times");

    // replay: a 300-token stretch repeated verbatim fires at n = 546
    std::vector<int> stretch;
    for (int i = 0; i < 300; ++i) stretch.push_back(1000 + i);
    LoopGuard g;
    size_t fired_at = 0;
    for (size_t j = 0; j < 600 && !fired_at; ++j)
        if (g.push(stretch[j % 300]))
            fired_at = j;
    check(fired_at == 546, "replay fires at n = 546 (got " + std::to_string(fired_at) + ")");

    // the loop guard ends a run of one token when the 32-token window repeats
    LoopGuard r;
    size_t run_end = 0;
    for (size_t j = 0; j < 100 && !run_end; ++j)
        if (r.push(5))
            run_end = j;
    check(run_end == 35, "constant run ends at token 35 (got " + std::to_string(run_end) + ")");

    // script guard
    check(
        latin_script_language("en") && latin_script_language("en-US") &&
            latin_script_language("pt_BR") && latin_script_language("FIL"),
        "latin languages");
    check(
        !latin_script_language("und") && !latin_script_language("hi") &&
            !latin_script_language("en-US-x") && !latin_script_language(""),
        "non-latin languages");
    check(
        script_guard("en", false) && script_guard("de", false) && !script_guard("ja", false),
        "served prompts guard Latin-script output languages");
    check(
        !script_guard("en", true) && !script_guard("de", true),
        "a client instruction sets the output language: no guard");
    check(
        is_latin_script(" Hello, world 42!") && is_latin_script("Ça, é, ß, ő") &&
            is_latin_script("€ — “”"),
        "latin text passes");
    check(
        !is_latin_script("हिन्दी") && !is_latin_script("தமிழ்") && !is_latin_script("Привет") &&
            !is_latin_script("中文"),
        "other scripts fail");
    check(!is_latin_script(std::string("\xE0\xA4", 2)), "partial UTF-8 fails");
    check(!is_latin_script("\xEF\xBF\xBD") && !is_latin_script("a\xEF\xBF\xBD"), "U+FFFD fails");

    // language guard (not NeMo): a chunk the model translated into English, or wrote in Latin
    // letters, instead of the requested language
    const std::string english =
        "You waited quite a long time to let us know, didn't you? I thought about calling you a "
        "few times, but with the time difference that was hard.";
    const std::string dutch =
        "En wat zei ze toen? Ze vond het jammer dat we zo laat belden. Jij hebt ook wel heel lang "
        "gewacht om iets te laten weten. Ik heb een paar keer gedacht om te bellen, maar met het "
        "tijdsverschil was dat lastig. Morgen gaan we samen naar de markt en daarna eten we bij "
        "oma.";
    check(off_language(english, "nl") && off_language(english, "nl-BE"), "English for Dutch");
    check(!off_language(dutch, "nl"), "Dutch for Dutch");
    check(!off_language(dutch + " You know what I mean?", "nl"), "an English phrase in Dutch text");
    check(!off_language(english, "en") && !off_language(english, "en-GB"), "English for English");
    check(off_language(english, "ja") && off_language(english, "ja_JP"), "English for Japanese");
    check(
        !off_language("では、明日の会議は十時に始めます。よろしくお願いします。", "ja"),
        "Japanese for Japanese");
    const std::string hindi =
        "आज हम एक नई film के बारे में बात करेंगे. ये कहानी थोड़ी सी emotional है, लेकिन पूरा परिवार "
        "साथ में बैठकर देख सकता है.";
    check(!off_language(hindi, "hi"), "Hindi with English words for Hindi");
    check(
        off_language(
            "Aaj hum ek nayi film ke baare mein baat karenge. Yeh kahani thodi si emotional hai.",
            "hi"),
        "romanised Hindi for Hindi");
    check(
        !off_language(english, "") && !off_language(english, "und") && !off_language(english, "xx"),
        "no usable language: no verdict");
    check(!off_language("Okay, thank you, I know what you mean.", "nl"), "too short for a verdict");
    check(!off_language("", "ja") && !off_language("1, 2, 3!", "ja"), "no letters: no verdict");

    if (failures)
        return 1;
    std::printf("salm_guards: all checks passed\n");
    return 0;
}
