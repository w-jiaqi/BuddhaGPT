import sys, buddhagpt.retrieval as R
hits = R.get_top("什么是空性？", k=3)
for h in hits:
    print(f"[{h['id']}] {h['text'][:80]}…\n")
