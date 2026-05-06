# Paper Finder

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

## Preview
First search some keyword for papers you would like to find. The dataset is heavily weighted towards physics papers.

<img width="1907" height="857" alt="image" src="https://github.com/user-attachments/assets/58c18eaf-6176-4a2a-8b64-08d7558834e6" /><br>

A keyword search is performed to show the top results for your search query.

<img width="1891" height="853" alt="image" src="https://github.com/user-attachments/assets/da95421c-b182-48b4-8122-514bdaa5d37e" /><br>

When viewing a paper recommendations will be loaded for papers that have high keyword similarity with the current paper. This is done using our trained
model.

<img width="1886" height="857" alt="image" src="https://github.com/user-attachments/assets/65b83242-5ab5-40c7-b84e-24cfff6a467b" /><br>
