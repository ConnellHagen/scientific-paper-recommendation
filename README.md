# Paper Finder (Flask + Jinja)

Using precomputed embeddings for scientific papers, this demo shows the search and "you may like" recommendations for papers.

## Quick start

After cloning, first [download the dataset and embeddings](https://drive.google.com/file/d/1JtnBUwks4OCNF8x6eRny0IuJ9tBRYEYE/view?usp=sharing). Unzip and place the
files in the root directory of this project.

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py --build-cache
python app.py
```

Open http://127.0.0.1:5000

## Dataset

By default the app reads:

```
arxiv-metadata-oai-snapshot-002.json
```

## Notes

- This is just a demo, and the locally run server is a bit slow. Give up to 20 seconds to wait for a page/recommendations to load before assuming something went wrong.

- GPT 5.2 Codex was used for help with caching and loading of data to greatly improve response times, and with general website styling
