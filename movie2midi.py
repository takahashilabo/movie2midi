"""動画ファイルのピアノ演奏をMIDIに変換するツール (movie2midi)"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """ffmpegで動画から音声をWAVに抽出する"""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", "44100",
            "-ac", "1",
            "-f", "wav",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ffmpegエラー:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def transcribe_basic_pitch(
    audio_path: Path,
    output_dir: Path,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length: float,
) -> Path:
    """basic-pitch (汎用多楽器モデル) でMIDIに変換する"""
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    predict_and_save(
        audio_path_list=[audio_path],
        output_directory=output_dir,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
    )

    midi_files = list(output_dir.glob("*.mid")) + list(output_dir.glob("*.midi"))
    if not midi_files:
        print("MIDIファイルが生成されませんでした", file=sys.stderr)
        sys.exit(1)
    return midi_files[0]


def transcribe_piano(audio_path: Path, output_path: Path) -> Path:
    """piano_transcription_inference (ピアノ専用モデル) でMIDIに変換する"""
    import librosa
    from piano_transcription_inference import PianoTranscription

    print("モデルをロード中 (初回は自動ダウンロード)...")
    transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)

    print("ピアノ音声を解析中...")
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    transcriptor.transcribe(audio, str(output_path))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="動画ファイルのピアノ演奏をMIDIに変換します"
    )
    parser.add_argument("input", nargs="?", default="a.mkv", help="入力動画ファイル (デフォルト: a.mkv)")
    parser.add_argument("-o", "--output", help="出力MIDIファイルパス (デフォルト: 入力ファイルと同じ名前で .mid)")
    parser.add_argument(
        "--model",
        choices=["basic-pitch", "piano"],
        default="basic-pitch",
        help="使用するモデル (デフォルト: basic-pitch)\n"
             "  basic-pitch: 汎用多楽器モデル (Spotify)\n"
             "  piano:       ピアノ専用高精度モデル (Kong et al.)",
    )
    # basic-pitch 専用オプション
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="[basic-pitch] 音符のオンセット検出しきい値 0.0〜1.0 (デフォルト: 0.5)",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="[basic-pitch] 音符のフレーム検出しきい値 0.0〜1.0 (デフォルト: 0.3)",
    )
    parser.add_argument(
        "--min-note-length",
        type=float,
        default=58.0,
        help="[basic-pitch] 最小音符長さ (ミリ秒, デフォルト: 58.0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".mid")

    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"モデル: {args.model}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        audio_path = tmp / "audio.wav"

        print("音声を抽出中...")
        extract_audio(input_path, audio_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.model == "piano":
            print("ピアノ音声をMIDIに変換中 (piano_transcription_inference)...")
            transcribe_piano(audio_path, output_path)
        else:
            print("ピアノ音声をMIDIに変換中 (basic-pitch)...")
            midi_file = transcribe_basic_pitch(
                audio_path=audio_path,
                output_dir=tmp,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                minimum_note_length=args.min_note_length,
            )
            midi_file.rename(output_path)

    print(f"完了: {output_path}")


if __name__ == "__main__":
    main()
