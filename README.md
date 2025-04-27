### Run unit tests


```sh
cd cs336-basics
uv run pytest
```
### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
cd cs336-basics
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Train model

```
cd cs336-basics
uv run cs336_basics/train_lm.py train --config tinystories.yaml
```


