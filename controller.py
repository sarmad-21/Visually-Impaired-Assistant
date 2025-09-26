from wakeup import wait_for_wakeword
from stt_system import stt
from ocr import text_recognition
from vision import analyze_environment, create_llava_prompt
from llava_runner import run_llava
from tts_system import speak

def main():
    while True:
        wait_for_wakeword()  # wait for wake up command 'Hey Vision'
        speak("I'm listening. Say a command.")  # acknowledge wakeup work was heard
        command = stt().lower().strip() # capture voice command and transcribe
        print(f"Command: {command}")

        if "read" in command or "paper" in command:
            speak("Reading the paper now.")
            text = text_recognition()
            speak(text)

        elif "around" or "infront" or "surroundings" in command:
            speak("Analyzing your surroundings.")
            env = analyze_environment()
            prompt = create_llava_prompt(env)
            response = run_llava(prompt)
            speak(response)

        elif "exit" in command or "stop" in command:
            speak("Goodbye.")
            break

        else:
            speak("I didn’t understand. Please say something like 'what’s around me' or 'read this paper'.")

if __name__ == "__main__":
    main()
