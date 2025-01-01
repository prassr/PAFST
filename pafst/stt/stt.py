import os
import json
from pathlib import Path
from tqdm import tqdm

from collections import defaultdict

import librosa
import whisper
from faster_whisper import WhisperModel
from whisper.tokenizer import LANGUAGES

from pydub import AudioSegment

from pafst.utils import write_json
from pafst.datasets import Dataset


whisper_model = {key: None for key in whisper._MODELS}

def whisper_stt(
        audio: Path,
        model_size='base',
        language=None,
):
    """
        Use the Whipser[https://github.com/openai/whisper] STT model to extract text.
        If there is no gpu or low performance, use the base model.

        Args:
            audio (Data): Audio data.
            model_size (str): Size of the whisper model.
            language (str): Language of the audio file to run STT.

        Return:
            str: Text in an audio file.

        """
    global whisper_model
    if model_size not in whisper_model.keys():
        raise ValueError(
            "[!] Invalid model size selected. Please choose one of the following sizes: tiny, base, small, medium, large.")

    if not whisper_model[model_size]:
        whisper_model[model_size] = whisper.load_model(model_size)

    result = whisper_model[model_size].transcribe(str(audio))

    return result

def process_segments(raw_segments):
    segments = []
    text=""
    for segment_chunk in raw_segments:
        chunk={}
        chunk["start"]=segment_chunk.start
        chunk["end"]=segment_chunk.end
        chunk["text"]=segment_chunk.text
        segments.append(chunk)
        text+=chunk["text"]+" "

    return segments, text

def segment_audio(audio_file, segments, output_path):
    """
    segments audio given a list of dict containing start time, end time
    """
    audio_data = []
    audio = AudioSegment.from_file(str(audio_file))
    
    for i, chunk in enumerate(segments):
        path = 'segment-%002d.wav' % (i,)
        path = (Path(output_path) / Path(path)).resolve()

        start_time_ms = chunk["start"] * 1000  #  in milliseconds
        end_time_ms = chunk["end"] * 1000  # in milliseconds
        segment = audio[start_time_ms:end_time_ms]

        segment.export(path, format="wav") 

        audio_data.append({
            "segment_path":str(path), 
            "audio_filepath": os.path.abspath(str(audio_file)),
            "start_time": chunk["start"], 
            "end_time": chunk["end"] ,
            "text": chunk["text"]
        })

    return audio_data


def STT(
        dataset: Dataset,
        model_size='tiny',
        vad=True,
        language=None,
        compute_type="int8",
        faster=False
        
):
    """
    Read the audio files in the dataset, and use the stt function to extract text.
    Save the extracted text in the form of a json file.

    Args:
        dataset (Dataset): Audio dataset Class
        output_format (str): Output format, Defaults is json (json or txt)
        model_size (str): Size of the whisper model.
        language (str): Language of the audio file to run STT.
        vad (bool) = True: Performs vad, saves the audio files on timestamps detected by vad.
        compute_type (str) = "int8": Loads the model in "int8" quantization
        faster (bool) = False: if True uses faster whisper
    Returns:
        List[Dict]: List of the dictionary containing audio files and processed files, language detected by whisper and timestamps if vad set to True.

    """

    if language and language not in LANGUAGES:
            raise ValueError(
            f"[!] This language is not supported. Please select one of the language codes below\n{LANGUAGES}")

    audios = dataset.audios
    stt_dict = defaultdict(dict)

    output_path = dataset.output_path

    if faster:
        model=WhisperModel(model_size, compute_type=compute_type)
        if language:
            options = dict(language=language, beam_size=5, best_of=5)
        else:
            options = dict(beam_size=5, best_of=5)
        transcribe_options=dict(task="transcribe", **options)
    
    audio_data=[]
    audio_main=[]
    bar = tqdm(audios,
               total=len(audios),
               leave=True,
               )
    for audio in bar:
        if faster:
            raw_segments, info=model.transcribe(str(audio), **transcribe_options)
            segments, text=process_segments(raw_segments)
            language=info.language
        else:
            result=whisper_stt(audio, model_size)
            segments, text, language = result["segments"], result['text'], result["language"]
        audio_main.append({
            "asr_text": text,
            "audio_filepath": str(audio),
            "language": language
        })

        if vad:
            a_data = segment_audio(audio, segments, output_path)
            audio_data.extend(a_data)

    write_json(output_path/"asr.json", audio_main)
    if vad:
        write_json(output_path/"whisper_vad.json", audio_data)
        return audio_data
    return audio_main
    
