from pathlib import Path

from tqdm import tqdm

from pafst.utils import write_json

from pafst.datasets import Dataset
from audio_separator.separator import Separator


def separator(
        dataset: Dataset
):
    separator = Separator(output_dir=dataset.output_path)
    separator.load_model()

    audios = dataset.audios
    audio_data = []

    bar = tqdm(audios,
               total=len(audios),
               leave=True,
               )

    for audio in bar:
        output_files = separator.separate(audio)
        target_path = Path(dataset.output_path) / Path(audio.name)

        # inst unlink
        (Path(dataset.output_path) / Path(output_files[0])).unlink()

        # save vocal
        if target_path.exists():
            target_path.unlink()
        Path(Path(dataset.output_path) / Path(output_files[1])).rename(target_path)

        audio_data.append({
                "separator_path":str(target_path),
                "audio_filepath": str(audio)
            })

    write_json(dataset.output_path/"separator.json", audio_data)

    return audio_data
