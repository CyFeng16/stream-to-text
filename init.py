import subprocess
from pathlib import Path

import hcvalues as hc
from func import paddle_asr_infer
from func import paddle_text_infer

if __name__ == "__main__":

    Path(hc.DATA_DIR).mkdir(exist_ok=True)
    init_dl = subprocess.run(
        f"wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav -O {hc.INIT_AUDIO_LOC}",
        shell=True,
    )
    if init_dl.returncode != 0:
        raise Exception("Failed to download the initial audio.")

    result_1 = paddle_asr_infer(Path(hc.INIT_AUDIO_LOC))
    result_2 = paddle_text_infer(result_1[-1])
    print(f"Result: {result_2}\nInitialization Success!")
    Path(hc.INIT_AUDIO_LOC).unlink()
