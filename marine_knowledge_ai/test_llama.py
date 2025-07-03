import subprocess

# Your question
user_prompt = "What is the pressure rating of the compressor?"

# Start the model process
process = subprocess.Popen(
    [r"C:\Users\gener\AppData\Local\Programs\Ollama\ollama.exe", "run", "gemma3"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    errors="replace"
)

# Send the prompt and get the output
stdout, stderr = process.communicate(input=user_prompt)

print("=== MODEL RESPONSE ===")
print(stdout.strip())

if stderr.strip():
    print("\n=== ERRORS ===")
    print(stderr.strip())
