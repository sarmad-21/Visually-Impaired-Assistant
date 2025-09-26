import whisper
import sounddevice as sd
import numpy as np
import wave

model = whisper.load_model("tiny")

def record_audio(filename="recorded.wav", duration=5, samplerate=16000):
    print("Recording, speak now...")
    recording = sd.rec(
        int(samplerate * duration), #total number of samples to record (16000 Hz * 10 sec)
        samplerate=samplerate,
        channels=1,
        dtype=np.int16  # record audio as 16 bit integer
    )
    sd.wait() #wait for recording to finish
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1) #mono
        wf.setsampwidth(2) #2 bytes = 16 bits
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    print("Saved audio to file")
    return filename

def stt():
        audio_file = record_audio()
        print('Transcribing audio...')
        result = model.transcribe(audio_file)
        print(result["text"])
        return result["text"]

if __name__ == "__main__":
    stt()
