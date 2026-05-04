# Paper Finder (Flask + Jinja)

Using a Python trained model on keywords for scientific papers, this demo shows the search and "you may like" recommendations for papers.

## Quick start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Dataset

By default the app reads:

```
arxiv-metadata-oai-snapshot-002.json
```

## Notes

On the first time that you open a paper, it may be a bit slower than any other
time. This is because the model hasn't been cached yet, and if this were running
in production it would be.
