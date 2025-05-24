from lxml import etree
from opencc import OpenCC
import json, pathlib, re, itertools, tqdm

RAW   = pathlib.Path("data/raw/cbeta_p5")
OUT_T = pathlib.Path("data/processed/cbeta_zh_trad.jsonl")
OUT_S = pathlib.Path("data/processed/cbeta_zh_simp.jsonl")
OUT_T.parent.mkdir(parents=True, exist_ok=True)

to_simp = OpenCC("t2s").convert
CHUNK_SIZE = 180

def local(tag):
    return tag.rpartition('}')[2]

def iter_passages(xml_path, chunk_size=CHUNK_SIZE):
    buf, first_id = [], None
    work_id = xml_path.stem 
    for ev, el in etree.iterparse(xml_path, events=("start", "end")):
        name = local(el.tag)

        if ev == "start" and name == "lb":
            first_id = f"{work_id}_{el.get('n')}"

        # paragraph text
        if ev == "end" and name == "p":
            txt = "".join(el.itertext()).strip()
            if txt:
                buf.append(txt)
                if sum(len(t) for t in buf) >= chunk_size:
                    last_id = first_id or f"{work_id}_unknown"
                    yield f"{first_id}-{last_id}", " ".join(buf)
                    buf, first_id = [], None
            el.clear()

    if buf:
        last_id = first_id or f"{work_id}_tail"
        yield f"{first_id}-{last_id}", " ".join(buf)

def main():
    files = list(RAW.rglob("T*/*.xml"))
    with OUT_T.open("w") as ft, OUT_S.open("w") as fs:
        for xml in tqdm.tqdm(files, desc="CBETA"):
            for pid, text in iter_passages(xml):
                rec = {"id": pid, "work": xml.stem, "trad": text}
                ft.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rec["simp"] = to_simp(text)
                fs.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
