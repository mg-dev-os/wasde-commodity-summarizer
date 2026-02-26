"""Summarize commodity commentary using OpenAI, Groq, or Google Gemini."""

from openai import OpenAI

from src.config_loader import (
    get_config,
    get_gemini_api_key,
    get_groq_api_key,
    get_openai_api_key,
)


def _get_openai_compatible_client():
    """Return OpenAI-compatible client for OpenAI, or Groq."""
    config = get_config()
    provider = (config.get("llm") or {}).get("provider", "openai")

    if provider == "groq":
        api_key = get_groq_api_key()
        if not api_key:
            return None
        groq = (config.get("llm") or {}).get("groq") or {}
        base_url = groq.get("base_url", "https://api.groq.com/openai/v1")
        return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    # openai
    api_key = get_openai_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _call_gemini(system: str, user: str, model_name: str, max_tokens: int, temperature: float) -> str:
    """Call Google Gemini API. Returns response text or raises."""
    import google.generativeai as genai

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/app/apikey")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"{system}\n\n{user}"
    config = {"max_output_tokens": max_tokens, "temperature": temperature}
    response = model.generate_content(prompt, generation_config=config)
    if not response or not response.text:
        raise RuntimeError("Gemini returned empty response")
    return response.text.strip()


def summarize_commentary(commodity_name: str, commentary: str) -> str:
    """
    Call configured LLM to summarize the given commentary for one commodity.
    Returns the summary text or an error message string.
    """
    config = get_config()
    provider = (config.get("llm") or {}).get("provider", "openai")
    openai_cfg = config.get("openai", {})
    groq_cfg = (config.get("llm") or {}).get("groq") or {}
    gemini_cfg = (config.get("llm") or {}).get("gemini") or {}

    if provider == "openai" and not get_openai_api_key():
        return "Error: OPENAI_API_KEY is not set. Add it to .env, or use llm.provider: groq or gemini (free cloud)."
    if provider == "groq" and not get_groq_api_key():
        return "Error: GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys and add to .env"
    if provider == "gemini" and not get_gemini_api_key():
        return "Error: GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/app/apikey and add to .env"

    # Treat very short or empty commentary like "no section" (same as Poultry when no text extracted)
    commentary_stripped = (commentary or "").strip()
    if len(commentary_stripped) < 50:
        return "(No commentary extracted for this section.)"

    # Truncate to avoid 413 / token limit (e.g. Groq 6k TPM; keep request under limit)
    max_chars = openai_cfg.get("max_commentary_chars", 12000)
    if len(commentary_stripped) > max_chars:
        commentary_stripped = commentary_stripped[:max_chars] + "\n\n[... truncated for length ...]"

    system = (
        "You are an analyst summarizing USDA WASDE report commentary. "
        "Output a concise, clear summary (2–5 sentences) of the key points: "
        "supply, demand, trade, stocks, and price outlook. Use plain language."
    )
    user = f"Summarize the following commentary for **{commodity_name}**:\n\n{commentary_stripped}"

    max_tokens = openai_cfg.get("max_tokens", 2048)
    temperature = openai_cfg.get("temperature", 0.3)

    if provider == "gemini":
        try:
            model_name = gemini_cfg.get("model", "gemini-1.5-flash")
            return _call_gemini(system, user, model_name, max_tokens, temperature)
        except Exception as e:
            msg = f"Error from Gemini: {e!s}"
            if "API_KEY" in str(e) or "api_key" in str(e).lower():
                msg += " Get a free key at https://aistudio.google.com/app/apikey"
            return msg

    client = _get_openai_compatible_client()
    if client is None:
        return "Error: Could not create LLM client. Set the API key for the chosen provider (see config.yaml)."

    if provider == "groq":
        model = groq_cfg.get("model", "llama-3.1-8b-instant")
    else:
        model = openai_cfg.get("model", "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()
    except Exception as e:
        err_str = str(e).lower()
        # Request too large (e.g. Groq 413) -> same style as Poultry: short plain message
        if "413" in str(e) or "request too large" in err_str or "token" in err_str and "limit" in err_str:
            return "(No commentary could be summarized for this section due to length limits. Please refer to the PDF for key points on supply, demand, and prices.)"
        labels = {"groq": "Groq", "openai": "OpenAI"}
        provider_label = labels.get(provider, provider)
        return f"Error from {provider_label}: {e!s}"


def summarize_all_commodities(sections: dict[str, str]) -> dict[str, str]:
    """
    Summarize each commodity section. sections: commodity_name -> commentary text.
    Returns commodity_name -> summary (or error message).
    """
    result: dict[str, str] = {}
    for name, text in sections.items():
        if not (text and text.strip()):
            result[name] = "(No commentary extracted for this section.)"
            continue
        result[name] = summarize_commentary(name, text)
    return result
