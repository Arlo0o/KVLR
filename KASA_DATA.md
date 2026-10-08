KASA anonymous data link:

https://drive.google.com/drive/folders/14vHj7IB-pA09QhrYT7D06Ei6A6md3z8k?usp=sharing

Expected local layout after download:

```text
<repo-root>/data/kasa/
  annotations/
  videos/
    knotting/
    needleGrasping/
    needlePuncture/
```

Google Drive contents:

- `annotations.zip`
- `knotting.zip`
- `needleGrasping.zip`
- `needlePuncture.zip`

Suggested extraction:

```bash
# Run from <repo-root>, the directory containing README.md.
mkdir -p data/kasa/annotations data/kasa/videos
unzip /path/to/annotations.zip -d data/kasa/annotations
unzip /path/to/knotting.zip -d data/kasa/videos
unzip /path/to/needleGrasping.zip -d data/kasa/videos
unzip /path/to/needlePuncture.zip -d data/kasa/videos
```
