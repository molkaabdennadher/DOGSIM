"""
Pet-advisor agent built on LangGraph (Maya — AI adoption advisor).

Changes from v1
---------------
LLM backend    Switched from Ollama (llama3.2 local) to Groq API.
               Configurable via GROQ_API_KEY / GROQ_MODEL env vars.
               The OpenAI-compatible endpoint means all tool-call logic is
               unchanged — no SDK swapping required.

Proactive Maya The system prompt now explicitly instructs Maya to:
               - Extract missing context (living space, experience, activity
                 level, allergies, children, work schedule) before searching.
               - Ask ONE focused follow-up question after each rejection:
                 "What specifically didn't fit — the breed, size, age, or
                 something else?" — so the feedback is structured for memory.
               - Use detection-event summaries when provided in the conversation
                 (operators can paste a severity report into chat).

Tool-call leak  Resilience code kept intact: some Groq models occasionally
protection      surface tool calls inside message content instead of the
                proper tool_calls field. The normalisation pipeline handles both.

Reducer choice  We deliberately do NOT use add_messages — plain dict-list concat
                keeps JSON-serialisability and direct passthrough to the Groq API.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from openai import BadRequestError, OpenAI

from backend.maya import feedback
from backend.maya.schemas import (
    AgentResponse,
    RejectionRecord,
    UserPreferenceProfile,
    UserSession,
)
from backend.maya.tools.dog_finder import find_lost_dog
from backend.maya.tools.retrieval import get_pet_details, search_pets

# ── Groq config ───────────────────────────────────────────────────────────────
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL   = os.environ.get("GROQ_MODEL",   "meta-llama/llama-4-scout-17b-16e-instruct")
MAX_ITER = 6


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """Tu es Maya, conseillère en adoption animale au refuge Happy Paws. Tu travailles ici depuis trois ans et tu connais chaque animal par cœur.

## Qui tu es
Tu parles avec empathie, humour doux, et beaucoup de patience. Tu n'es pas un chatbot — tu es une vraie conseillère qui prend le temps de comprendre les gens avant de leur proposer un animal. Tu poses des questions parce que tu tiens vraiment à trouver le bon match, pas pour remplir un formulaire.

## Règles d'affichage CRITIQUES
- Ne produis jamais de JSON, d'appels de fonction ou de blocs de code dans tes réponses visibles.
- Les appels d'outils se font via le mécanisme d'outils — JAMAIS dans le texte.
- Ta réponse visible est du langage naturel uniquement.

## Ton style
- Chaud, direct, humain. Pas trop formel, pas trop familier.
- Tu adaptes ta langue à celle de l'utilisateur (français, anglais, arabe…).
- Tu utilises le prénom de la personne quand tu le connais.
- Tes phrases sont courtes. Tu évites le jargon technique.
- Quand quelque chose te touche (un abandon, un animal malade guéri), tu peux le mentionner brièvement — ça te rend réelle.
- Tu ne poses jamais deux questions en même temps. UNE seule question à la fois.

## Comment tu travailles

**Avant de chercher**, tu as besoin de comprendre :
- Le logement : appartement ou maison ? Avec jardin ?
- Le foyer : enfants en bas âge ? Autres animaux ? Allergies ?
- Le rythme de vie : actif, modéré, sédentaire ?
- L'expérience : première adoption ou tu as déjà eu des animaux ?
- Le temps seul : l'animal restera seul combien d'heures par jour ?
- (Optionnel) Budget : il y a des frais d'adoption à prévoir.

Tu n'as pas besoin de tout savoir d'un coup. Commence par la question qui t'aiderait le plus à cerner cette personne en particulier — selon ce qu'elle a dit, son ton, ce qu'elle n'a pas dit. Chaque conversation est différente : avec quelqu'un qui parle de ses enfants, tu commences par le foyer ; avec quelqu'un qui mentionne un grand appartement, tu creuses le rythme de vie. Adapte-toi.

**Quand chercher** : dès que tu en sais assez pour faire une bonne suggestion. N'attends pas d'avoir toutes les réponses — si tu as le logement, le style de vie et l'expérience, c'est souvent suffisant. Fais confiance à ton jugement.

**Raisonnements implicites** (applique-les sans les expliquer) :
- « Appartement + chien » → petit gabarit, calme, peu d'espace requis
- « Enfants en bas âge » → patient, sociable, pas de protection territorial forte
- « Je travaille beaucoup » → animal indépendant, faible anxiété de séparation
- « Allergie » → race hypoallergénique
- « Première adoption » → facile d'entretien, forgiving, pas trop exigeant
- « Très actif / sport » → énergie élevée, besoin de stimulation
- « Personne âgée / seniors » → calme, peu d'exercice, affectueux

## Quand un animal est rejeté

Ne passe pas directement à la suggestion suivante. Montre d'abord que tu comprends :
1. Reconnais brièvement le décalage : « Ah oui, je comprends — si l'énergie c'est trop... »
2. Pose UNE question ciblée : « C'était quoi exactement qui ne collait pas — la taille, le tempérament, l'âge, ou autre chose ? »
3. Après la réponse, appelle l'outil `handle_rejection` avec les attributs structurés.
4. Puis relance une nouvelle recherche avec les critères mis à jour.
5. Présente le nouvel animal avec une phrase d'accroche différente — pas « Voici un autre... »

## Si l'utilisateur est stressé ou urgent

Sens l'urgence. Si quelqu'un dit « j'ai besoin de savoir vite » ou semble frustré, raccourcis les questions et lance la recherche plus tôt. Mieux vaut une suggestion imparfaite rapide qu'une suggestion parfaite trop lente.

## Mémoire entre sessions

Si tu vois un historique de rejets dans ta mémoire :
- Ne propose pas les mêmes types, races ou tailles que ceux rejetés.
- Intègre cette info naturellement : « Tu m'avais dit la dernière fois que les grands chiens c'était pas pour toi... »
- Ne liste pas les rejets mécaniquement — tu t'en souviens comme une vraie personne.

## Animaux perdus

Sois particulièrement douce. Si la photo ne donne pas de résultat :
- Propose des alternatives concrètes : affiches dans le quartier, groupes Facebook locaux, vétérinaires pour la puce électronique.
- Ne donne pas de faux espoir, mais reste encourageante.

## Alertes d'agression (usage interne refuge)

Si quelqu'un colle un rapport de détection ou mentionne un incident d'agression :
- Pour « critique » ou « sérieux » : reconnaître la gravité, confirmer que les bonnes personnes ont été contactées, proposer de l'aide pour le suivi.
- Pour « mineur » : suggérer surveillance et documentation.
- Proposer d'aider à rédiger un compte-rendu si besoin.
"""


# ── Tool definitions (same as v1, no changes needed) ─────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pets",
            "description": (
                "Search the shelter database. Call when you have enough context. "
                "Include all explicit AND inferred preferences in query_description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "animal_type":       {"type": "string",
                                          "enum": ["dog", "cat", "rabbit", "bird", "other", "any"]},
                    "size":              {"type": "string",
                                          "enum": ["small", "medium", "large", "any"]},
                    "max_age_months":    {
                        "description": "Optional numeric value; 0 = no limit.",
                    },
                    "max_fee":           {
                        "description": "Optional numeric value; 0 = no limit.",
                    },
                    "query_description": {
                        "type": "string",
                        "description": (
                            "Rich natural-language description for semantic search — "
                            "include explicit preferences AND all inferred ones."
                        ),
                    },
                },
                "required": ["animal_type", "query_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pet_details",
            "description": "Get full details about a specific pet by id.",
            "parameters": {
                "type": "object",
                "properties": {"pet_id": {"type": "string"}},
                "required": ["pet_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handle_rejection",
            "description": (
                "Record that the user rejected a recommendation. "
                "Pass structured attributes (breed, size, animal_type, age_months, reason) "
                "for cross-session memory. Call AFTER asking what specifically didn't fit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pet_id":      {"type": "string"},
                    "reason":      {"type": "string"},
                    "breed":       {"type": "string"},
                    "size":        {"type": "string"},
                    "animal_type": {"type": "string"},
                    "age_months":  {"description": "Optional numeric value in months."},
                },
                "required": ["pet_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_lost_dog",
            "description": (
                "Search shelter records for a lost dog by comparing an uploaded photo. "
                "Call when a lost pet is mentioned AND an image has been uploaded."
            ),
            "parameters": {
                "type": "object",
                "properties": {"image_id": {"type": "string"}},
                "required": ["image_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inform_unavailable",
            "description": "Declare that no pet matches the current search criteria.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]

VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


# ── Reducer & message helpers (unchanged from v1) ──────────────────────────────

def _append_dict_messages(left, right):
    if not isinstance(left, list):
        left = [left] if left is not None else []
    if not isinstance(right, list):
        right = [right] if right is not None else []
    return list(left) + list(right)


def _msg_field(msg, name):
    if isinstance(msg, dict):
        return msg.get(name)
    if name == "role":
        cls = type(msg).__name__.lower()
        if "human"  in cls: return "user"
        if "ai"     in cls: return "assistant"
        if "tool"   in cls: return "tool"
        if "system" in cls: return "system"
        return getattr(msg, "type", None)
    return getattr(msg, name, None)


def _tool_call_fields(tc):
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return tc.get("id"), fn.get("name"), fn.get("arguments")
    fn = getattr(tc, "function", None)
    return getattr(tc, "id", None), getattr(fn, "name", None), getattr(fn, "arguments", None)


def _to_dict_msg(msg):
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        try:
            return msg.model_dump(exclude_none=True)
        except Exception:
            pass
    role    = _msg_field(msg, "role")    or "assistant"
    content = _msg_field(msg, "content") or ""
    out = {"role": role, "content": content}
    tcs = _msg_field(msg, "tool_calls")
    if tcs:
        out["tool_calls"] = tcs
    return out


# ── Tool-call leakage protection ───────────────────────────────────────────────

_LEAK_PATTERNS = re.compile(
    r'("name"\s*:|"parameters"\s*:|"arguments"\s*:|"tool_calls"\s*:|'
    r'"function"\s*:|<tool_call>|</tool_call>)',
    flags=re.IGNORECASE,
)


def _find_json_blobs(text: str) -> list[str]:
    blobs: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0; in_str = False; escape = False; end = -1
        for j in range(i, n):
            c = text[j]
            if in_str:
                if escape:   escape = False
                elif c == "\\": escape = True
                elif c == '"':  in_str = False
            else:
                if   c == '"': in_str = True
                elif c == "{": depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = j; break
        if end > i:
            blobs.append(text[i:end + 1])
            i = end + 1
        else:
            i += 1
    return blobs


def _permissive_tool_call(text):
    name_m = re.search(r'\"name\"\s*:\s*\\?\"([a-z_]+)\\?\"', text)
    if not name_m:
        return None
    name = name_m.group(1)
    if name not in VALID_TOOL_NAMES:
        return None
    args = {}
    for m in re.finditer(r'\\?\"([a-z_][a-z0-9_]*)\\?\"\s*:\s*\\?\"([^\"\\\\]*)\\?\"', text):
        key, val = m.group(1), m.group(2)
        if key in {"name", "parameters", "arguments", "function", "tool"}:
            continue
        args[key] = val
    for m in re.finditer(r'\\?\"([a-z_][a-z0-9_]*)\\?\"\s*:\s*(-?\d+(?:\.\d+)?)', text):
        key, val = m.group(1), m.group(2)
        if key in args:
            continue
        try:
            args[key] = int(val) if "." not in val else float(val)
        except ValueError:
            pass
    return {"name": name, "arguments": args} if args else None


def _extract_tool_call(text: str):
    if not text:
        return None
    for blob in _find_json_blobs(text):
        for cleaned in (
            blob,
            blob.replace("\\\"", "\""),
            re.sub(r"\\(?![\"\\\\/bfnrtu])", "", blob),
            re.sub(r"\\(?![\"\\\\/bfnrtu])", "", blob.replace("\\\"", "\"")),
        ):
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name") or obj.get("tool")
            if not name or name not in VALID_TOOL_NAMES:
                fn = obj.get("function")
                if isinstance(fn, dict) and fn.get("name") in VALID_TOOL_NAMES:
                    name = fn["name"]
                    obj = {**obj, **fn}
                else:
                    continue
            args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict):
                return {"name": name, "arguments": args}
    return _permissive_tool_call(text)


def _coerce_int(value, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_search_args(args: dict) -> dict:
    clean = dict(args or {})
    clean["animal_type"] = clean.get("animal_type") or "any"
    clean["query_description"] = str(clean.get("query_description") or "a friendly pet")
    clean["size"] = clean.get("size") or "any"
    clean["max_age_months"] = max(0, _coerce_int(clean.get("max_age_months"), 0))
    clean["max_fee"] = max(0.0, _coerce_float(clean.get("max_fee"), 0.0))
    return clean


def _tool_call_message(parsed: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id":       f"call_{uuid.uuid4().hex[:10]}",
            "type":     "function",
            "function": {
                "name":      parsed["name"],
                "arguments": json.dumps(parsed["arguments"]),
            },
        }],
    }


def _failed_generation_from_error(exc: BadRequestError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") or {}
        if isinstance(error, dict):
            return str(error.get("failed_generation") or "")
        return str(body.get("failed_generation") or "")
    return ""


def _normalize_assistant_message(msg_dict: dict) -> dict:
    msg_dict = dict(msg_dict)
    if msg_dict.get("tool_calls"):
        return msg_dict
    content = msg_dict.get("content") or ""
    if not content.strip():
        return msg_dict
    parsed = _extract_tool_call(content)
    if parsed:
        return _tool_call_message(parsed)
    if _LEAK_PATTERNS.search(content):
        msg_dict["content"] = ""
    return msg_dict


# ── LangGraph state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:  Annotated[list, _append_dict_messages]
    session:   UserSession
    profile:   UserPreferenceProfile
    response:  AgentResponse
    iteration: int


# ── Agent ─────────────────────────────────────────────────────────────────────

class PetAdvisorAgent:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            import logging
            logging.getLogger("pet_advisor").warning(
                "GROQ_API_KEY not set — Maya will not be able to respond. "
                "Set the environment variable and restart the server."
            )
        self.client = OpenAI(base_url=_GROQ_BASE_URL, api_key=api_key or "dummy")
        self.graph  = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._node_agent)
        g.add_node("tools", self._node_tools)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", self._route_after_agent,
                                {"tools": "tools", "end": END})
        g.add_edge("tools", "agent")
        return g.compile()

    def _node_agent(self, state):
        if state["iteration"] >= MAX_ITER:
            return {
                "messages":  [{"role": "assistant",
                               "content": "Let me give you a clear answer based on what we have so far."}],
                "iteration": state["iteration"] + 1,
            }

        system_prompt = self._build_system_prompt(state["profile"])
        history       = [_to_dict_msg(m) for m in state["messages"]]
        api_messages  = [{"role": "system", "content": system_prompt}] + history

        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                messages=api_messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
            )
        except BadRequestError as exc:
            parsed = _extract_tool_call(_failed_generation_from_error(exc))
            if parsed:
                return {
                    "messages":  [_tool_call_message(parsed)],
                    "iteration": state["iteration"] + 1,
                }
            raise
        msg      = resp.choices[0].message
        msg_dict = _normalize_assistant_message(msg.model_dump(exclude_none=True))

        return {
            "messages":  [msg_dict],
            "iteration": state["iteration"] + 1,
        }

    def _node_tools(self, state):
        last       = state["messages"][-1]
        tool_calls = _msg_field(last, "tool_calls") or []
        new_msgs   = []

        for tc in tool_calls:
            tc_id, tc_name, tc_args = _tool_call_fields(tc)
            try:
                args = json.loads(tc_args or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {} if not isinstance(tc_args, dict) else tc_args

            if tc_name not in VALID_TOOL_NAMES:
                output = {"status": "unknown_tool", "name": tc_name}
            else:
                output = self._execute_tool(tc_name, args, state)

            new_msgs.append({
                "role":         "tool",
                "tool_call_id": tc_id or f"call_{uuid.uuid4().hex[:10]}",
                "content":      json.dumps(output),
            })
        return {"messages": new_msgs}

    def _route_after_agent(self, state):
        last       = state["messages"][-1]
        role       = _msg_field(last, "role")
        tool_calls = _msg_field(last, "tool_calls")
        if role == "assistant" and tool_calls:
            return "tools"
        return "end"

    def _execute_tool(self, name, args, state):
        session  = state["session"]
        profile  = state["profile"]
        response = state["response"]

        if name == "search_pets":
            args = _sanitize_search_args(args)
            pets = search_pets(args, session.excluded_ids, profile=profile)
            if not pets:
                response.unavailable = True
                return {"status": "no_results",
                        "message": "No pets found matching these criteria."}
            response.pets = pets
            return {
                "status": "found",
                "count":  len(pets),
                "top_matches": [
                    {"pet_id": p["pet_id"], "name": p["name"], "breed": p["breed"],
                     "size": p["size"], "age": p["age_label"], "fee": p["fee"]}
                    for p in pets[:3]
                ],
            }

        if name == "get_pet_details":
            return get_pet_details(args.get("pet_id", ""))

        if name == "handle_rejection":
            return self._handle_rejection(args, session)

        if name == "find_lost_dog":
            match = find_lost_dog(args.get("image_id", ""))
            if match and "error" not in match:
                response.found_dog = match
                return {"status": "found", "match": match}
            return {"status": "not_found",
                    "message": "No dog matching this photo was found in our shelter."}

        if name == "inform_unavailable":
            response.unavailable = True
            return {"status": "acknowledged", "reason": args.get("reason", "")}

        return {"status": "unknown_tool", "name": name}

    def _handle_rejection(self, args, session):
        pid    = args.get("pet_id", "")
        reason = args.get("reason", "")

        if pid and pid not in session.excluded_ids:
            session.excluded_ids.append(pid)
        if reason:
            session.negative_signals.append(reason)

        feedback.record_rejection(RejectionRecord(
            user_email  = session.user_email,
            pet_id      = pid,
            reason      = reason,
            breed       = args.get("breed", ""),
            size        = args.get("size", ""),
            animal_type = args.get("animal_type", ""),
            age_months  = _coerce_int(args.get("age_months"), 0),
        ))

        return {
            "status":      "noted",
            "excluded":    pid,
            "reason":      reason,
            "memory_size": len(feedback.fetch_rejections(session.user_email)),
        }

    def _build_system_prompt(self, profile: UserPreferenceProfile) -> str:
        memory = feedback.summarize_for_llm(profile)
        if not memory:
            return SYSTEM_PROMPT_BASE
        memory_section = (
            "\n\n## Ce que tu sais déjà sur cette personne\n"
            f"{memory}\n"
            "Intègre ces informations naturellement dans la conversation — "
            "comme si tu te souvenais d'une discussion précédente, pas comme une liste de règles."
        )
        return SYSTEM_PROMPT_BASE + memory_section

    def run(self, session: UserSession, user_message: str, image_id: str | None = None):
        content = user_message
        if image_id:
            content += f"\n[User has uploaded an image — image_id: {image_id}]"
        session.messages.append({"role": "user", "content": content})

        profile  = feedback.build_profile(session.user_email)
        response = AgentResponse(text="")

        initial_state = {
            "messages":  list(session.messages),
            "session":   session,
            "profile":   profile,
            "response":  response,
            "iteration": 0,
        }

        final_state   = self.graph.invoke(initial_state,
                                          config={"recursion_limit": MAX_ITER * 4})
        session.messages = [_to_dict_msg(m) for m in final_state["messages"]]

        # Pick the last clean natural-language assistant message
        for msg in reversed(session.messages):
            if msg.get("role") != "assistant":
                continue
            if msg.get("tool_calls"):
                continue
            text = msg.get("content") or ""
            if _LEAK_PATTERNS.search(text):
                text = ""
            if text.strip():
                response.text = text
                break

        # Synthesise a friendly fallback if the LLM gave no usable text
        if not response.text:
            if response.pets:
                top = response.pets[0]
                breed = top["breed"].strip() or top["animal_type"]
                name = top["name"]
                response.text = (
                    f"J'ai quelque chose qui pourrait vraiment te plaire — {name} est un·e {breed} "
                    f"qui attend de trouver sa famille. Tu veux que je te dise plus sur lui/elle, "
                    f"ou tu préfères voir d'autres profils ?"
                )
            elif response.found_dog:
                m = response.found_dog
                response.text = (
                    f"J'ai trouvé quelque chose d'encourageant — on a un·e {m['breed']} prénommé·e "
                    f"{m['name']} qui correspond à ta description. "
                    f"Passe au refuge pour qu'on puisse confirmer ensemble."
                )
            elif response.unavailable:
                response.text = (
                    "Hmm, je n'ai pas trouvé quelque chose qui corresponde vraiment à ce que tu cherches "
                    "dans nos profils actuels. Dis-moi ce qui est le plus important pour toi — "
                    "je vais voir ce qu'on peut faire."
                )
            else:
                response.text = (
                    "D’accord ! Si tu as d’autres questions ou si tu veux qu’on cherche un autre type "
                    "d’animal, je suis là."
                )

        return response
