"""Provider-agnostic, grounded explanations for classifier results."""
from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
DEFAULT_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"
CLASS_FILES = {"cardiac": "cardiac.txt", "chroniclung": "chroniclung.txt", "normal": "normal.txt", "pleural": "pleural.txt", "tb": "tb.txt"}


def _context(prediction: str | None) -> str:
    docs = {p.name: p.read_text(encoding="utf-8") for p in KNOWLEDGE_DIR.glob("*.txt")}
    key = re.sub(r"[^a-z]", "", (prediction or "").lower())
    selected = [docs[CLASS_FILES[key]]] if key in CLASS_FILES and CLASS_FILES[key] in docs else []
    selected.append(docs.get("system_overview.txt", ""))
    return "\n\n---\n\n".join(text for text in selected if text)


def _cache_file(value: str) -> Path:
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    return cache / f"{hashlib.sha256(value.encode()).hexdigest()}.json"


def _api_key(api_url: str) -> str | None:
    if "groq.com" in api_url.lower():
        return os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
    return os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")


def _post_json(api_url: str, payload: dict, api_key: str) -> dict:
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json", "User-Agent": "Mozilla/5.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _vision_observation(image_base64: str, predicted_class: str | None) -> str:
    """Ask a vision model for a bounded second opinion, never a diagnosis."""
    if not image_base64:
        return "Vision comparison was not requested."
    api_url = os.environ.get("LLM_API_URL", DEFAULT_API_URL)
    api_key = _api_key(api_url)
    if not api_key:
        return "Vision comparison unavailable because the LLM API key is not configured."
    image_url = image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"
    prompt = (
        "You are an experimental second-opinion image classifier, not a medical diagnostician. "
        f"The primary GATv2 classifier predicted {predicted_class}. Inspect the image independently and choose exactly one label from: "
        "Cardiac, ChronicLung, Normal, Pleural, TB, or Uncertain. "
        "Return JSON only with keys vision_class, vision_confidence (0 to 1), observed_regions (short list), and note (one short sentence). "
        "Do not prescribe treatment. Do not claim certainty. This result is a comparison signal and must not override the primary classifier."
    )
    payload = {"model": os.environ.get("VISION_MODEL", DEFAULT_VISION_MODEL), "temperature": 0.0,
               "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}]}
    try:
        data = _post_json(api_url, payload, api_key)
        content = data["choices"][0]["message"]["content"].strip()
        # Vision models sometimes wrap JSON in a markdown code fence even when
        # asked for JSON. Extract the object defensively before parsing.
        if content.startswith("``"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(content)
        return json.dumps(parsed, ensure_ascii=False)
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = ""
        return f"Vision comparison unavailable: HTTP {error.code}; model={payload['model']}; provider={body}"
    except Exception as error:
        return f"Vision comparison unavailable: {error}; model={payload['model']}"


def vision_comparison(image_base64: str | None, predicted_class: str | None) -> dict:
    """Return the raw second-opinion result for the API/UI comparison panel."""
    if not image_base64:
        return {"status": "not_requested"}
    raw = _vision_observation(image_base64, predicted_class)
    try:
        parsed = json.loads(raw)
        return {"status": "ok", **parsed}
    except (TypeError, ValueError):
        return {"status": "unavailable", "note": raw}


def explain_result(prediction: str | None, confidence: float | None, review_needed: bool, question: str | None = None, focus_regions: list[str] | None = None, image_base64: str | None = None, vision_note: str | None = None) -> str:
    context = _context(prediction)
    if not context:
        return (
            f"Detected: {prediction or 'Unknown'} (Confidence: {(confidence or 0) * 100:.1f}%)\n\n"
            "Key sign: Not available in the retrieved context\n\n"
            "Region to inspect: Not available in the retrieved context\n\n"
            "Why: The knowledge base does not contain a class-specific explanation for this label."
        )
    user_question = question or f"Explain the model output '{prediction}' with confidence {(confidence or 0):.1%}. Review needed: {review_needed}."
    # Version the cache when the safety wording changes so old explanations
    # with less precise language are not served again.
    focus_regions = focus_regions or []
    vision_note = vision_note or (_vision_observation(image_base64, prediction) if image_base64 else "Vision comparison was not requested.")
    cache_file = _cache_file(f"rag-v7-vision-compare|{prediction}|{confidence}|{review_needed}|{focus_regions}|{vision_note}|{user_question}")
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8")).get("answer")
        if cached:
            return cached
    except (OSError, ValueError, TypeError):
        pass
    api_url = os.environ.get("LLM_API_URL", DEFAULT_API_URL)
    # Prefer the provider-specific key when using Groq. This avoids an old,
    # invalid generic LLM_API_KEY shadowing a valid GROQ_API_KEY in Railway.
    api_key = _api_key(api_url)
    if not api_key:
        raise RuntimeError("RAG is not configured: set LLM_API_KEY in the backend environment.")
    system = ("You are a radiologist-facing explanation layer for a chest X-ray classification model. "
              "The reader is a radiologist or clinician, not a general patient. You do not diagnose; explain the model output using only the retrieved context. "
              "Use ONLY the supplied CONTEXT to describe signs, regions, and reasoning. Never invent a radiological sign, region, or explanation. "
              "You may compare the supplied VISION SECOND CLASSIFICATION with the primary classifier. State agreement or disagreement plainly, but never treat either output as a confirmed diagnosis or let the vision result replace the primary classifier. "
              "Do not include generic patient disclaimers or 'consult a doctor' language. Do not suggest medications, dosages, treatment protocols, or urgent-care instructions. "
              "Always output exactly this four-part structure and nothing else. Do not use Markdown bold, bullet points, or code fences. Keep every field on one line.\n\n"
              "Detected: {predicted_class} (Confidence: {confidence_percent}%)\n\n"
              "Key sign: {signs from context}\n\n"
              "Region to inspect: {region from context}\n\n"
              "Why: {one or two sentences connecting the supplied model-focused regions to the context's Why this region field. "
              "Start with 'The model attention was concentrated broadly in ...' when a focus region is supplied. "
              "Do not call a generic chest region an organ and do not infer a clinical finding from attention alone.}\n\n"
              "Use plain radiological language. Do not add any other headers or commentary.\n\n"
              f"predicted_class: {prediction}\nconfidence_percent: {(confidence or 0) * 100:.1f}\n"
              f"model_focused_regions: {', '.join(focus_regions) if focus_regions else 'not available'}\n\nCONTEXT:\n{context}")
    system += f"\n\nVISION SECOND CLASSIFICATION (lower authority than the primary classifier; report agreement or disagreement):\n{vision_note}"
    payload = json.dumps({"model": os.environ.get("LLM_MODEL", DEFAULT_MODEL), "temperature": 0.1,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_question}]}).encode()
    request = urllib.request.Request(api_url, data=payload,
                                     headers={
                                         "Authorization": f"Bearer {api_key}",
                                         "Content-Type": "application/json",
                                         "Accept": "application/json",
                                         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                                     }, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        answer = data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as error:
        try:
            provider_body = error.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            provider_body = ""
        if error.code in {401, 403}:
            raise RuntimeError(
                "Groq rejected this request (HTTP %s). Check the Railway key/model "
                "permissions. Provider response: %s" % (error.code, provider_body)
            ) from error
        raise RuntimeError(f"RAG provider request failed: HTTP {error.code}") from error
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as error:
        raise RuntimeError(f"RAG provider request failed: {error}") from error
    cache_file.write_text(json.dumps({"answer": answer}), encoding="utf-8")
    return answer


def answer_followup(prediction: str | None, confidence: float | None, focus_regions: list[str], question: str) -> str:
    """Answer a follow-up using the current result and retrieved class context."""
    context = _context(prediction)
    if not context:
        return "No class-specific context is available for this result."
    api_url = os.environ.get("LLM_API_URL", DEFAULT_API_URL)
    api_key = _api_key(api_url)
    if not api_key:
        raise RuntimeError("RAG is not configured: set GROQ_API_KEY in the backend environment.")
    prompt = (
        "You are a radiologist-facing follow-up explanation assistant. Answer only from the supplied context and current model result. "
        "Do not diagnose, prescribe treatment, or invent image findings. Explain model behaviour, class facts, confidence, or graph focus only. "
        "If the question is outside the context, say that it is not covered. Keep the answer under 100 words.\n\n"
        f"Current prediction: {prediction}; confidence: {(confidence or 0) * 100:.1f}%; "
        f"graph focus: {', '.join(focus_regions) if focus_regions else 'not available'}\n\n"
        f"CONTEXT:\n{context}"
    )
    payload = {"model": os.environ.get("LLM_MODEL", DEFAULT_MODEL), "temperature": 0.1,
               "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question}]}
    try:
        data = _post_json(api_url, payload, api_key)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as error:
        raise RuntimeError(f"RAG follow-up failed: {error}") from error


def chat_assistant(message: str, conversation_history: list[dict] | None = None) -> dict:
    """Conversational assistant for medical chest X-ray inquiries with domain restrictions.
    
    Accepts free-form chat while strictly maintaining domain and safety restrictions.
    Can ask clarifying questions and engage in multi-turn conversation.
    
    Args:
        message: User's current message
        conversation_history: List of previous messages in format [{"role": "user|assistant", "content": "..."}, ...]
    
    Returns:
        dict with keys: "response" (str), "can_answer" (bool), "clarifying_question" (bool)
    """
    if conversation_history is None:
        conversation_history = []

    # Do not send meaningless one-character input to the LLM. With the full
    # knowledge base in its context, a model can otherwise try to guess an
    # intended topic and return unrelated performance terminology.
    if len(re.sub(r"[^a-zA-Z0-9]", "", message)) < 2:
        return {
            "response": "Please ask a specific question about chest X-ray findings, TB, cardiac, pleural, chronic lung, or normal findings.",
            "can_answer": False,
            "clarifying_question": True,
            "role": "assistant",
        }
    
    # Load all available knowledge base content
    docs = {p.name: p.read_text(encoding="utf-8") for p in KNOWLEDGE_DIR.glob("*.txt")}
    knowledge_context = "\n\n---\n\n".join([docs.get("system_overview.txt", ""), docs.get("cardiac.txt", ""), 
                                             docs.get("chroniclung.txt", ""), docs.get("normal.txt", ""), 
                                             docs.get("pleural.txt", ""), docs.get("tb.txt", "")])
    
    api_url = os.environ.get("LLM_API_URL", DEFAULT_API_URL)
    api_key = _api_key(api_url)
    if not api_key:
        raise RuntimeError("RAG is not configured: set LLM_API_KEY in the backend environment.")
    
    system_prompt = (
        "You are a medical AI assistant specialized in chest X-ray analysis and interpretation. "
        "You engage in natural, multi-turn conversation with radiologists and medical professionals.\n\n"
        
        "DOMAIN RESTRICTIONS (STRICT):\n"
        "1. ONLY answer questions about chest X-ray findings, pathologies, and analysis within your knowledge base.\n"
        "2. ONLY discuss: Cardiac Pathology, Chronic Lung Disease, Normal findings, Pleural Pathology, Tuberculosis (TB).\n"
        "3. DO NOT diagnose patients, prescribe medications/dosages, or suggest treatment plans.\n"
        "4. DO NOT provide urgent care instructions or medical advice to patients.\n"
        "5. DO NOT discuss topics outside medical imaging (e.g., coding, philosophy, politics).\n"
        "6. If a question is outside your domain, politely decline and redirect to the knowledge base topics.\n\n"
        "7. Never guess what an unclear word, acronym, typo, or very short message means. Ask the user to clarify instead.\n"
        "8. Do not mention FDR, FPR, domain shift, external datasets, accuracy figures, or other model-performance details unless the user explicitly asks about that exact model-performance topic.\n\n"
        
        "ASSISTANT BEHAVIOR:\n"
        "- Engage naturally and conversationally, not as a rigid Q&A system.\n"
        "- Ask clarifying questions when the user's intent is unclear.\n"
        "- Reference specific findings from the knowledge base when relevant.\n"
        "- Discuss model limitations only when the user asks about reliability, confidence, performance, or limitations.\n"
        "- Provide educational context about why certain regions matter in X-ray interpretation.\n"
        "- Suggest related topics or follow-up questions that may be helpful.\n\n"
        
        "SAFETY GUARDRAILS:\n"
        "- Never claim the system is a substitute for radiologist review.\n"
        "- Always remind users that borderline or uncertain cases need qualified radiologist review.\n"
        "- Do not use absolutes like 'this is definitely TB' — use probabilistic language.\n"
        "- Never invent explanations, definitions, or related topics that are not directly asked for.\n\n"
        
        "KNOWLEDGE BASE (use ONLY this content to answer):\n"
        f"{knowledge_context}"
    )
    
    # Build message history for API call
    messages = []
    # Add conversation history (limit to last 10 turns for context window)
    if conversation_history:
        messages.extend(conversation_history[-10:])
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "temperature": 0.3,  # Slightly higher than RAG for conversational flow
        "messages": [{"role": "system", "content": system_prompt}] + messages
    }
    
    try:
        data = _post_json(api_url, payload, api_key)
        response_text = data["choices"][0]["message"]["content"].strip()
        
        # Determine response type
        can_answer = "i cannot" not in response_text.lower() and "not covered" not in response_text.lower()
        clarifying_question = any(mark in response_text for mark in ["?", "Could you clarify", "What do you mean"])
        
        return {
            "response": response_text,
            "can_answer": can_answer,
            "clarifying_question": clarifying_question,
            "role": "assistant"
        }
    except Exception as error:
        raise RuntimeError(f"Chat assistant failed: {error}") from error
