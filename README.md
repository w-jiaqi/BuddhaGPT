# BuddhaGPT

**BuddhaGPT** 是一个面向中文用户的检索增强型语言模型（RAG）。它先在佛教大藏经（CBETA）向量索引中检索最相关经典段落，再用大语言模型 (Qwen-7B-Instruct) 输出。

## 环境安装

```bash
git clone https://github.com/w-jiaqi/BuddhaGPT.git
cd BuddhaGPT
conda create -n buddha-gpt python=3.11 -y
conda activate buddha-gpt
pip install -r requirements.txt
```

### LFS 大文件

```bash
git lfs install
git lfs pull
```

---

### 运行

```bash
python rag/inference.py "如何对治嗔恨心？"
```

### 示例输出

```text
问：云何方便除嗔恚？答：与彼应作周旋，应思惟其功德恩、自业所作负债解脱，亲族自身罪过不应作意，自现苦诸根自性念灭…

简释：经文教我们先与怨敌周旋并修慈心，观对方功德而舍嗔；若无功德则起慈悲，久之嗔恚自灭。
```
