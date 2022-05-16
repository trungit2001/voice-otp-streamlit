from pydub import AudioSegment
from pydub.silence import split_on_silence

def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def convert_to_wav(inpath, outpath):
    sound = AudioSegment.from_file(inpath)
    sound.export(outpath, format="wav")

def segment(inpath, outpath):
    sound = AudioSegment.from_wav(inpath)

    chunks = split_on_silence (
        sound,
        min_silence_len = 300,
        silence_thresh = sound.dBFS - 16,
        keep_silence = 250,
    )

    for i, chunk in enumerate(chunks):
        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(chunk, -20.0)

        # Export the audio chunk with new bitrate.
        normalized_chunk.export(
            f'{outpath}/segment_{i}.wav',
            bitrate = "192k",
            format = "wav"
        )
