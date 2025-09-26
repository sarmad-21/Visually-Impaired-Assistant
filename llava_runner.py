import subprocess
from vision import analyze_environment, create_llava_prompt

def run_llava(prompt, image_path="object.jpg"):
    result = subprocess.run([
        "./build/bin/llama-llava-cli",
        "-m", "models/llava-phi-2.Q4_K_M.gguf",
        "--mmproj", "models/llava-phi2-3b-mmproj-model-f16.gguf",
        "--image", image_path,
        "-p", prompt
    ], capture_output=True, text=True)

    return result.stdout.strip()

if __name__ == "__main__":
    env_info = analyze_environment()
    prompt = create_llava_prompt(env_info)
    print("Prompt for LLaVA:", prompt)
    llava_response = run_llava(prompt)
    print("LLaVA Response:", llava_response)
