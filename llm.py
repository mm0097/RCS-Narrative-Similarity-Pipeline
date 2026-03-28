import ollama


def gen_ollama(prompt: str, model_name: str, system_instruction: str, json_schema: dict,
               temperature: float = 0.0) -> str | None:
    """Call Ollama chat with JSON schema enforcement. Returns response content or None on error."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            format=json_schema,
            options={"temperature": temperature},
        )
        return response.message.content
    except Exception as e:
        print(f"Ollama chat error: {e}")
        return None


def gen_ollama_embeddings(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed a list of texts using Ollama. Returns list of embedding vectors."""
    response = ollama.embed(model=model_name, input=texts)
    return response.embeddings
