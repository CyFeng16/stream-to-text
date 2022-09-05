from func import get_stream_audio
from func import paddle_asr_infer
from func import paddle_text_infer

if __name__ == "__main__":

    # input stream url
    url = input("Please input the stream url: ")
    # get the stream audio
    stream_audio_chunks = get_stream_audio(url)
    stream_audio_chunks.sort()
    # asr and punctuation
    asr_results = []
    punc_results = []

    for no, fp in enumerate(stream_audio_chunks):
        print(f"[{no+1}/{len(stream_audio_chunks)}]{fp} is processing...")
        result_1 = paddle_asr_infer(fp)
        asr_results.append(result_1[-1])
    asr_results = "".join(asr_results)
    for i in range(len(asr_results) // 511 + 1):
        result_2 = paddle_text_infer(asr_results[i * 511 : (i + 1) * 511])
        punc_results.append(result_2)
    punc_results = "".join(punc_results)
    print(f"Final Result: {punc_results}")
