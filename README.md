# movie2midi

ピアノ演奏の動画ファイルから MIDI データを生成するツールです。  
動画から音声を抽出し、AI モデルで音声解析による MIDI 変換を行います。

## 必要なもの

- Python 3.11 以上
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/)

## セットアップ

```bash
# 基本依存パッケージのインストール
uv sync

# basic-pitch の依存パッケージ追加
uv pip install "setuptools<81" "basic-pitch[onnx]"
```

ピアノ専用モデル (`--model piano`) を使う場合は追加で:

```bash
uv pip install "piano-transcription-inference" "torch" --index-url https://download.pytorch.org/whl/cpu
```

> 初回実行時にモデルファイル（約165MB）が自動ダウンロードされます。

## 使い方

```bash
# 基本 (入力ファイルと同じ名前で .mid を生成)
uv run python movie2midi.py input.mkv

# 出力先を指定
uv run python movie2midi.py input.mkv -o output.mid

# ピアノ専用モデルで変換（より高精度）
uv run python movie2midi.py input.mkv -o output.mid --model piano
```

## モデルの選択

| モデル | 特徴 | 用途 |
|---|---|---|
| `basic-pitch` (デフォルト) | 汎用・多楽器対応 (Spotify) | ピアノ以外の楽器も含む場合 |
| `piano` | ピアノ専用・高精度 (Kong et al.) | ピアノ演奏に特化、倍音誤検出が少ない |

## オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--model` | `basic-pitch` | 使用モデル (`basic-pitch` または `piano`) |
| `--onset-threshold` | `0.5` | \[basic-pitch\] 音符の発音しきい値（0〜1、低いほど多く検出） |
| `--frame-threshold` | `0.3` | \[basic-pitch\] 音符の持続しきい値（0〜1） |
| `--min-note-length` | `58` | \[basic-pitch\] 最小音符長さ（ミリ秒） |

### basic-pitch の精度調整

検出される音符が少ない場合はしきい値を下げてみてください。

```bash
uv run python movie2midi.py input.mkv --onset-threshold 0.3 --frame-threshold 0.2
```

## 仕組み

1. **音声抽出** — ffmpeg で動画から音声のみ WAV に変換
2. **AI 推論** — 選択したモデルが音程・発音・持続を解析
   - `basic-pitch`: CNN ベースの汎用モデル
   - `piano`: CRNN ベースのピアノ専用モデル（ベロシティ・サステイン考慮）
3. **MIDI 生成** — 検出した音符をタイミング・長さつきで MIDI に書き出し

## 対応フォーマット

ffmpeg が対応している動画・音声フォーマットであれば入力として使用できます（MKV, MP4, AVI, WAV, MP3 など）。

## ライセンス

MIT
