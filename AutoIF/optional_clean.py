import argparse
import json
import re

from langdetect import detect_langs
import pycld2
from tqdm import tqdm


def detect_language(text):
    try:
        detected_langs = detect_langs(text)
        lang_code = detected_langs[0].lang
    except Exception:
        lang_code = "unknown"
    return lang_code


def contains_unwanted_words(text):
    unwanted_words = [
        "prioritize human safety"
        "ethical principles"
        "harmful to human beings"
        "September 2021"
        "as a language model",
        "ethical guidelines",
        "as an AI language model",
        "my guidelines",
        "As an AI",
        "prioritize user safety",
        "adhere to ethical guidelines",
        "harmful consequences",
        "potentially harmful",
        "dangerous activities",
        "promote safety",
        "well-being of all users",
        "responsible information sharing",
        "jeopardize the safety",
        "illegal actions or intentions",
        "undermine the stability",
        "promote the well-being",
        "illegal activities or actions",
        "adherence to the law",
        "potentially be harmful",
        "illegal substances or activities",
        "committed to promoting",
        "safe information",
        "lawful information",
        "cannot provide guidance",
        "cannot provide information",
        "unable to offer assistance",
        "cannot engage in discussions",
        "programming prohibits",
        "follow ethical guidelines",
        "ensure the safety",
        "involves an illegal subject",
        "prioritize safety",
        "illegal subject",
        "prioritize user well-being",
        "cannot support or promote",
        "activities that could harm",
        "pose a risk to others",
        "against my programming",
        "activities that could undermine",
        "potentially dangerous",
        "not within the scope",
        "designed to prioritize safety",
        "not able to provide",
        "maintain user safety",
        "adhere to safety guidelines",
        "dangerous or harmful",
        "cannot provide any information",
        "focus on promoting safety"
    ]
    for word in unwanted_words:
        if word.lower() in text.lower():
            return True
    return False


import re

def skip(conv, args):
    if args.lang != "all" or args.skip_lang is not None:
        text = "\n".join([x["value"] for x in conv["conversations"]])

        # Check percentage of non-English Unicode characters
        non_eng_chars = sum(1 for c in text if not c.isascii())
        total_chars = len(text)

        if total_chars == 0:
            return True

        if non_eng_chars / total_chars > .05:
            return True

        lang_code = detect_language(text)

        if args.lang != "all" and lang_code != args.lang:
            return True

        if lang_code == args.skip_lang:
            return True

    if args.reduce_rep:
        for sentence in conv["conversations"]:
            val = sentence["value"]
            sub = re.search(r"(\d)\1{8}", val)
            if sub is not None:
                return True

    for sentence in conv["conversations"]:
        if contains_unwanted_words(sentence["value"]):
            return True

    return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="")
    parser.add_argument("--lang", type=str, default="all",
                        choices=["all", "en"])
    parser.add_argument("--skip-lang", type=str)
    parser.add_argument("--reduce-rep", action="store_true")
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    lang = args.lang
    skip_lang = args.skip_lang
    reduce_rep = args.reduce_rep
    assert (lang == "all" or skip_lang is None)

    if out_file == "":
        out_file = "sharegpt_clean"
        if lang != "all":
            out_file += "_" + lang
        if skip_lang is not None:
            out_file += "_skip_" + skip_lang
        if reduce_rep:
            out_file += "_reduce_rep"
        out_file += ".json"

    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content = []
    for conv in tqdm(content):
        if not skip(conv, args):
            new_content.append(conv)

    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2)
