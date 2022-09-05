import subprocess
from pathlib import Path
from typing import List

import paddle
from paddlespeech.cli.asr import ASRExecutor
from paddlespeech.cli.text import TextExecutor

import hcvalues as hc
from utils import get_audio_pathlib


def get_stream_audio(url: str) -> List[Path]:
    """
    Get the specific streaming audio, then get the audio chunks from the stream
    :param url: url of the stream
    :return: file paths of the audio chunks
    """

    print("Getting the streaming audio...")
    ytb_dl = subprocess.run(
        f"youtube-dl {url} -x --audio-format wav --audio-quality 0 --no-continue -o {hc.STREAM_AUDIO_LOC}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if ytb_dl.returncode != 0:
        raise Exception("Failed to download the streaming audio.")

    print("Converting the audio to 16KHz...")
    ffmpeg_ar = subprocess.run(
        f"ffmpeg -i {hc.STREAM_AUDIO_LOC} -ar 16k {hc.STREAM_16K_AUDIO_LOC}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if ffmpeg_ar.returncode != 0:
        raise Exception("Failed to convert the streaming audio to 16000Hz.")

    print(f"Segmenting the audio into {hc.STREAM_AUDIO_SEGMENT_TIME}s chunks...")
    ffmpeg_seg = subprocess.run(
        f"ffmpeg -i {hc.STREAM_16K_AUDIO_LOC} -f segment -segment_time {hc.STREAM_AUDIO_SEGMENT_TIME} -c copy {hc.DATA_DIR}%04d.wav",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if ffmpeg_seg.returncode != 0:
        raise Exception("Failed to segment the streaming audio.")

    print("Deleting the original audio...")
    Path(hc.STREAM_AUDIO_LOC).unlink()
    Path(hc.STREAM_16K_AUDIO_LOC).unlink()

    return get_audio_pathlib(hc.DATA_DIR, full_path=True)


def paddle_asr_infer(audio_fp: Path or str) -> List[str]:
    """
    Get the ASR result of the audio
    :param audio_fp: file path of the audio
    :return: List of audio name and ASR result, e.g. ["audio.wav", "你好"]
    TODO: use Ray (multi-processes) to speed up the inference
    """

    asr_executor = ASRExecutor()
    asr_result = asr_executor(
        model="conformer_online_wenetspeech",
        lang="zh",
        sample_rate=16000,
        config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
        ckpt_path=None,
        audio_file=audio_fp,
        force_yes=True,
        device=paddle.get_device(),
    )

    return [audio_fp.name, asr_result]


def paddle_text_infer(text: str) -> str:
    """
    Get punctuation of the text
    :param text: result of the ASR
    :return: text with punctuation
    """

    # Get the text from the ASR result
    # ATTENTION: text must be less than 512 characters according to the model limitation
    text_punc = TextExecutor()
    punc_result = text_punc(text=text)

    return punc_result
