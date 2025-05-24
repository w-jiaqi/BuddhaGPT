import torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from opencc import OpenCC
from rag.retrieval import get_top

cc = OpenCC("t2s")
LLM = "Qwen/Qwen2-7B-Instruct"

_tok, _model = None, None
def llm():
    global _tok, _model
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            LLM, trust_remote_code=True,
            device_map="auto", torch_dtype=torch.float16)
    return _tok, _model

SYSTEM = (
    "你是佛典问答助手，只做两步："
    "① 输出检索到的经文（转简体，截取约 160 字）；"
    "② 接着用 1–2 句简体写“简释：…”。除此之外什么都别写。"
)

def answer(query: str, k: int = 5, max_gen=96):
    top = get_top(query, k)
    if not top:
        return "未找到对应经文，请重新提问。"

    # 取首段并截长
    raw  = top[0]["text"].replace("\n", "")
    clip = raw[:160] + ("…" if len(raw) > 160 else "")
    # quote = f"[{top[0]['id']}] {cc.convert(clip)}"
    quote = cc.convert(clip)

    prompt = (
        f"<|system|>\n{SYSTEM}"
        f"<|user|>\n问题：{query}\n<|assistant|>\n{quote}\n\n简释："
    )

    tok, model = llm()
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp, max_new_tokens=max_gen, do_sample=False, temperature=0.0,
        repetition_penalty=1.1)
    gloss = tok.decode(out[0][inp["input_ids"].shape[1]:],
                       skip_special_tokens=True).split("。")[:2]
    gloss_txt = "。".join(s.strip() for s in gloss if s.strip()).rstrip("。") + "。"
    return f"{quote}\n\n简释：{gloss_txt}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python rag.py \"<提问>\"")
        sys.exit(0)
    print(answer(sys.argv[1]))
