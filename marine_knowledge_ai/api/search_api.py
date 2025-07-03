from flask import Flask, request, jsonify
from flask_cors import CORS
from hybrid_search import hybrid_search
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        print(f"🔍 Received query: {query}")
        results = hybrid_search(query, top_k=5)
        return jsonify(results)
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask')
def ask():
    question = request.args.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        print(f"💬 Asking LLM: {question}")
        results = hybrid_search(question, top_k=5)
        context = "\n\n".join([chunk['full_text'] for chunk in results])

        prompt = f"""
You are an intelligent technical assistant helping engineers understand and analyze documentation related to marine, industrial, and mechanical systems.

Your task is to generate a clear, comprehensive answer based only on the context provided below.

Follow these formatting rules:

1. Use <strong>Field:</strong> value for technical specifications and bullet-style details.
2. If the answer includes tabular data (e.g., multiple models or values), format it using proper HTML <table> tags.
3. Always insert <br> after every field/value pair for readability.
4. If possible, cite the <em>document name</em> and <em>page number</em> (if provided in metadata) for each fact or group of facts.
5. Prefer long, detailed answers over short summaries. Be factual, avoid hallucinations.
6. Be robust: handle both specific and open-ended questions gracefully.

Context:
{context}

Question: {question}

Answer (HTML-formatted):
"""

        process = subprocess.Popen(
            [r"C:\Users\gener\AppData\Local\Programs\Ollama\ollama.exe", "run", "gemma3"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        stdout, stderr = process.communicate(input=prompt)

        return jsonify({
            "answer": stdout.strip(),
            "sources": results,
            "llm_error": stderr.strip() if stderr.strip() else None
        })

    except Exception as e:
        print(f"❌ Ask failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
