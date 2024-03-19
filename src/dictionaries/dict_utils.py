import random
import os
from pathlib import Path
import time

libreoffice_fnames = ["fr_FR.dic", "it_IT.dic", "ro_RO.dic", "en_GB.dic", "nl_NL.dic"]
ccedict_fname = ["zh_ZH.dic"]


def load_libreoffice_dict(lang_code, path=""):
    """language code in the format fr_FR"""
    lang_fname = lang_code + ".dic"
    if lang_fname not in libreoffice_fnames:
        print(f"Cannot load {lang_fname} dictionary")
        return

    words = []
    if path == "":
        fpath = Path(os.getcwd()) / "src/dictionaries" / lang_fname
    else:
        fpath = Path(path) / lang_fname

    # print(os.getcwd())
    with open(fpath, "r") as fin:
        for line in fin.readlines():
            if "/" in line:
                word = line.split("/")[0]
                words.append(word.strip())
            else:
                words.append(line.strip())
    print(lang_code, len(words))
    return words


def remove_surnames(parsed_entries):
    for x in range(len(parsed_entries) - 1, -1, -1):
        if "surname " in parsed_entries[x]["english"]:
            if parsed_entries[x]["traditional"] == parsed_entries[x + 1]["traditional"]:
                parsed_entries.pop(x)


def load_cc_cedict(entry_type="traditional"):
    """return a list of words of entry_type from cc_cedict"""

    parsed_entries = []
    with open("zh_ZH.dic") as file:
        text = file.read()
        lines = text.split("\n")
        dict_lines = list(lines)
        for line in dict_lines:
            parsed = {}
            if line == "":
                continue
            line = line.rstrip("/")
            line = line.split("/")
            if len(line) <= 1:
                continue
            english = line[1]
            char_and_pinyin = line[0].split("[")
            characters = char_and_pinyin[0]
            characters = characters.split()
            traditional = characters[0]
            simplified = characters[1]
            # print(char_and_pinyin)
            pinyin = char_and_pinyin[1]
            pinyin = pinyin.rstrip()
            pinyin = pinyin.rstrip("]")
            parsed["traditional"] = traditional
            parsed["simplified"] = simplified
            parsed["pinyin"] = pinyin
            parsed["english"] = english
            parsed_entries.append(parsed)

    remove_surnames(parsed_entries)
    res = []
    for e in parsed_entries:
        res.append(e[entry_type])
    return res


def get_random_word(word_list):
    length = len(word_list)
    random.seed(time.time())
    idx = random.randint(0, length - 1)
    word = word_list[idx]
    while len(word) < 3:
        idx = random.randint(0, length - 1)
        word = word_list[idx]

    return word_list[idx]


def load_dictionary(lang_code):
    if lang_code == "zh_ZH":
        return load_cc_cedict()
    else:
        return load_libreoffice_dict(lang_code)


def main():
    dict_list = []
    dict_list.append(load_cc_cedict())
    dict_list.append(load_libreoffice_dict("it_IT"))
    dict_list.append(load_libreoffice_dict("fr_FR"))
    dict_list.append(load_libreoffice_dict("en_GB"))
    dict_list.append(load_libreoffice_dict("nl_NL"))
    dict_list.append(load_libreoffice_dict("ro_RO"))
    for d in dict_list:
        print(len(d), get_random_word(d))


if __name__ == "__main__":
    main()
