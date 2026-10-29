// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The /v1/audio/transcriptions bodies: verbose_json words carry a character unit's running text
// (display) and whether it follows the previous unit without a space (joins_previous), so a client
// can rebuild the transcript from the timed words as the srt and vtt formats do.
#include <iostream>
#include <string>

#include "http_server.h"
#include "json.h"
#include "types.h"

namespace asr = nemo_speech::asr;

#define CHECK(condition, what)                                        \
    do {                                                              \
        if (!(condition)) {                                           \
            std::cerr << "FAIL " << __LINE__ << ": " << what << '\n'; \
            return 1;                                                 \
        }                                                             \
    } while (0)

int
main() {
    asr::Result result;
    asr::Alternative alternative;
    alternative.transcript =
        "\xE5\xB0\x8F\xE6\x9E\x97\xE3\x81\xAB\xE5\x8F\xB1\xE3\x82\x89\xE3\x82\x8C\xE3\x82\x8B\xE3"
        "\x80\x82";
    asr::Word plain;  // a unit whose display is its word
    plain.word = "\xE5\xB0\x8F";
    plain.start_time = 0;
    plain.end_time = 200;
    asr::Word joined;  // carries a folded character and follows without a space
    joined.word = "\xE3\x81\xAB";
    joined.display = "\xE3\x81\xAB\xE5\x8F\xB1";
    joined.joins_previous = true;
    joined.start_time = 200;
    joined.end_time = 400;
    alternative.words = {plain, joined};
    result.alternatives.push_back(alternative);

    const auto body = nemo_speech::json::Value::parse(
        nemo_speech::http::transcript_response(result, "ja", "verbose_json"));
    const auto& words = body.at("words").array();
    CHECK(words.size() == 2, "two words");
    CHECK(words[0].at("word").string() == plain.word, "word keeps the unit");
    CHECK(!words[0].find("display"), "no display when it equals the word");
    CHECK(!words[0].find("joins_previous"), "no joins_previous for a spaced unit");
    CHECK(words[1].at("word").string() == joined.word, "word keeps the unit");
    const auto* display = words[1].find("display");
    CHECK(display && display->string() == joined.display, "display carries the running text");
    const auto* joins = words[1].find("joins_previous");
    CHECK(joins && joins->boolean(), "joins_previous marks an unspaced unit");

    const auto json = nemo_speech::json::Value::parse(
        nemo_speech::http::transcript_response(result, "ja", "json"));
    CHECK(!json.find("words"), "plain json has no words");
    std::cout << "transcript response: PASS\n";
    return 0;
}
