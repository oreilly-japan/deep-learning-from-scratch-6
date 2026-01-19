ゼロから作る Deep Learning ❻ —LLM編
=============================

<!-- [<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-5/images/cover.png" width="200px">](https://www.amazon.co.jp/dp/4814400594/) -->


書籍『[ゼロから作るDeep Learning ❻](https://www.amazon.co.jp/dp/4814401612)』（オライリー・ジャパン発行）のサポートサイトです。本書籍で使用するソースコードがまとめられています。


## ファイル構成

|フォルダ名 |説明                             |
|:--        |:--                              |
|`ch01`〜`ch09`|各章で使用するコード|
|`codebot`   |CodeBotで使用するコードやデータ |
|`storybot`   |StoryBotで使用するコードやデータ |
|`webbot`   |WebBotで使用するコードやデータ |
<!-- |`notebooks`   |1章〜6章までのコード（Jupyter Notebook形式）| -->

## Pythonと外部ライブラリ

ソースコードを実行するには下記のライブラリが必要です。

* NumPy
* Matplotlib
* PyTorch 2.x
* tqdm

※Pythonのバージョンは **3.10以上** を利用します。


## 環境構築

### uvを使う場合

[uv](https://docs.astral.sh/uv/)は高速なPythonパッケージマネージャーです。`uv.lock`により、OSを問わず同じ環境を再現できます。
```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh   # Mac/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

実行は`uv run`を使います。環境構築も自動で行われます。
```bash
uv run python ch01/01_char_tokenizer.py
```

### pipを使う場合
```bash
pip install -r requirements.txt
python ch01/01_char_tokenizer.py
```


## 実行方法

各章のフォルダへ移動して実行するか、親フォルダから実行します。
```bash
# 各章のフォルダ内で実行
$ cd ch01
$ python 01_char_tokenizer.py

# 親フォルダから実行
$ python ch02/10_gpt.py
```

uvを使う場合は`python`を`uv run python`に置き換えてください。

## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。


## 正誤表

本書の正誤情報は以下のページで公開しています。

https://github.com/oreilly-japan/deep-learning-from-scratch-6/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた方は、[japan@oreilly.co.jp](<mailto:japan@oreilly.co.jp>)までお知らせください。