import shutil
from pathlib import Path
import uuid
from datetime import datetime
from typing import Union, Dict, List

from pafst.datasets import Dataset
from pafst.denoisers import denoiser
from pafst.vad import vad
from pafst.diarization import diarization
from pafst.separator import separator
from pafst.stt import STT

class PAFST:
    """
        Make audio files into a dataset for TTS/STT.

        Args:
        path (str): Directory path with audio files.
        dataset_name (str, optional): Dataset name. Defaults to dataset_path's directory name.
        language (str, optional): Language using BCP 47 language tag. Defaults to 'en-us' (English)
        output_path (str): Output Directory. Defaults to './pafst_output'

        Example with quick start:
        p = PAFTS(
            path = 'your_audio_directory_path',
            output_path = 'output_path',
            hf_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"
        )
        p.run()

        """

    def __init__(
            self,
            path: str = None,
            dataset_name: str = None,
            language: str = None,
            output_path: str = 'pafst_output',
            hf_token: str = None
    ):

        self._hf_token=hf_token
        self.language=language
       
        self._dataset = Dataset(
            path=path,
            dataset_name=dataset_name,
            language=language,
            output_path=output_path
        )
        
    def separator(self):
        """
        input: 
            None, uses roformer to separate audio source from music source
        output: 
            List[Dict]]: This list contains the origianl file path and the processed audio path
        """
        audio_data=separator(self._dataset)
        return audio_data
    
    def vad(self, detector="webrtc", params: Dict={}):
        """
        input: 
            detector: "webrtc" (Optional) # uses "webrtcvad"
            params: {
                mode: 0-3 # aggressiveness, 0-mild, 3-very aggressive
                frame_duration: 10 or 20 or 30 # accepted duration in ms
                padding_duration: in milliseconds
            } (Optional)
        output: 
            List[Dict]]: This list contains the origianl file path and the processed audio path with vad timestamps
        """
        audio_data = []
        if detector == "webrtc":
            if params:
                audio_data = vad(self._dataset, **params)
            else:
                audio_data = vad(self._dataset)
        elif detector == "silero_vad":
            pass

        return audio_data

    def denoiser(self, processor="dfn"):
        """
        input: 
            processor: "dfn" (Optional)
                "dfn" is DeepFilterNet3, use "den" for facebook denoiser
        output: 
            List[Dict]]: This list contains the origianl file path and the processed audio path
        """
        audio_data = denoiser(self._dataset, processor=processor)
        return audio_data

    def diarization(self, hf_token=None):
        """
        input: 
            hf_token (Optional)
        output: 
            List[Dict]]: This list contains the origianl file path and the processed audio path
        """
        if hf_token:
            self._hf_token = hf_token

        audio_data = diarization(self._dataset, self._hf_token)
        return audio_data

    def stt(self, model_size='large-v3', vad=True, language="en", compute_type="float32"):
        """
        Uses faster_whisper.
        input:
            model_size = "large-v3" # 
            vad=True # performs Voice Activity Detection
            language="en"
            compute_type="float32" # the quantization size. float32 is no quantization at all
        output:
            List[Dict]]: This list contains the origianl file path and the processed audio path
        """
        if self.language:
            language=self.language

        audio_data = STT(self._dataset, model_size=model_size, 
                         vad=vad, language=language, compute_type=compute_type)
        return audio_data

    def _stage_process(self, process_function, *args, **kwargs):
        # Create unique temp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex
        temp_dir = Path.cwd() / f"temp_dir_{timestamp}_{unique_id}"
        temp_dir.mkdir(exist_ok=True)

        self._dataset.output_path = temp_dir
        process_function(self._dataset, *args, **kwargs)

        # Update dataset path
        self._dataset.path = temp_dir

        return temp_dir

    def run(self):
        """
        Perform separation, diarization and stt in one go.
        """

        # if not processor_functions:
        #     processor_functions = ["separator", "diarization", "stt"]
        # functions = {
        #     "separator": separator,
        #     "vad": vad,
        #     "denoiser": denoiser,
        #     "diarization": diarization,
        #     "stt": STT
        # }
        original_output = self._dataset.output_path

        # Stage 1: separator
        temp_dir1 = self._stage_process(separator)

        # Stage 2: diarization
        temp_dir2 = self._stage_process(diarization, hf_token=self._hf_token)

        # Stage 3: STT
        temp_dir3 = self._stage_process(STT)

        original_output.mkdir(exist_ok=True)

        for temp_dir in [temp_dir2, temp_dir3]:
            for file in temp_dir.rglob('*'):
                if file.is_file():
                    relative_path = file.relative_to(temp_dir)
                    destination = original_output / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file, destination)

        # Cleanup
        for temp_dir in [temp_dir1, temp_dir2, temp_dir3]:
            shutil.rmtree(temp_dir, ignore_errors=True)
