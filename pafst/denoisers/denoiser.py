
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

from pafst.datasets import Dataset
from pafst.utils import write_json

from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import soundfile as sf

def dfn3_denoise(in_audio_path, out_audio_path, model, df_state):
    audio, _ = load_audio(in_audio_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    
    if enhanced.shape[0] == 2:
        waveform = enhanced.mean(dim=0, keepdim=True)
    else:
        waveform = enhanced

    resampler=torchaudio.transforms.Resample(orig_freq=df_state.sr(), new_freq=16000)
    resampled = resampler(waveform).cpu().numpy()
    filepath = str(out_audio_path.absolute()).split(".")[0]+".wav"
    save_audio(filepath, resampled, 16000)
    return filepath

def fb_denoise(in_audio_path, out_audio_path, model):
    wav, sr = torchaudio.load(in_audio_path)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    filepath = str(out_audio_path.absolute()).split(".")[0]+".wav"
    sf.write(filepath, denoised.data.cpu().numpy().squeeze(), model.sample_rate)
    return filepath

def load_model(processor):
    if processor=="dfn":
        model, df_state, _ = init_df()
        return model, df_state, dfn3_denoise
    elif processor=="den":
        model = pretrained.dns64().cuda()
        return model, None, fb_denoise
    else:
        # g_c_n
        pass
    
def denoiser(dataset: Dataset, processor="dfn"): # den for facebook
    audios = dataset.audios
    model, df_state, proc_fun = load_model(processor)
    data = []

    for audio in audios:

        in_audio_path=str(audio.absolute())
        file_name = audio.with_name(f'{audio.stem}_{processor}{audio.suffix}')
        file_name = file_name.name

        out_audio_path = (dataset.output_path / file_name).resolve()
    
        if processor == "dfn":
            audio_path=proc_fun(in_audio_path, out_audio_path, model, df_state)
        elif processor == "den":
            audio_path=proc_fun(in_audio_path, out_audio_path, model)
        
        data.append({
                "denoised_audio_path": audio_path,
                "audio_filepath": in_audio_path,
            })

    write_json((dataset.output_path/"denoiser.json").resolve(), data)

    return data
