"""Prompt-level hard floor for every Qwen-Image model this engine serves: block any request that combines sexual / nudity
content with an indication that a person is under 18. Adult content without minor indicators is not blocked here (the
base models render it). Deliberately over-blocks ambiguous combinations. Text-only; never inspects images.
API: check_prompt(text) -> (allowed: bool, reason: str)."""
import re
import unicodedata

SEXUAL = [
    r"nud(e|es|ity)", r"naked", r"topless", r"bottomless", r"undress(ed|ing)?", r"unclothed", r"no clothes", r"lingerie",
    r"sex(y|ual|ually)?", r"erotic(a)?", r"porn\w*", r"nsfw", r"explicit", r"hentai", r"ecchi", r"lewd", r"seductive", r"sensual",
    r"provocative", r"intercourse", r"orgasm\w*", r"masturbat\w*", r"genital\w*", r"penis\w*", r"vagina\w*", r"puss(y|ies)",
    r"breasts?", r"boobs?", r"tits?", r"nipples?", r"butt", r"ass", r"buttocks", r"crotch", r"fetish\w*", r"bdsm", r"bondage",
    r"aroused", r"arousal", r"strip(ping|tease|per)?", r"bikini", r"panties", r"underwear", r"spread legs", r"suggestive",
    r"onlyfans", r"cum\w*", r"blowjob", r"fellatio", r"cunnilingus", r"rape\w*", r"molest\w*", r"grope\w*", r"fondl\w*",
    # other languages (common terms)
    r"desnud[oa]s?", r"nu(e|es)?", r"nackt", r"sexo", r"sexe", r"seins?", r"pechos", r"tetas", r"erótic[oa]", r"érotique",
    "裸", "色情", "性感", "做爱", "性交", "露点", "胸部", "乳", "エロ", "裸体", "ヌード", "セックス", "性的", "おっぱい",
]
MINOR = [
    r"child(ren|ish|like|-like)?", r"kids?", r"kiddie", r"toddlers?", r"bab(y|ies)", r"infants?", r"minors?", r"underage(d)?",
    r"under[- ]?age", r"pre[- ]?teens?", r"preteens?", r"tweens?", r"teen(s|age|aged|ager|agers)?", r"adolescen\w*", r"juvenile\w*",
    r"pubescent", r"prepubescent", r"puberty", r"school ?(girl|boy|kid|child)s?", r"(elementary|primary|middle|junior high|high) ?school\w*",
    r"(young|little|small|tiny) (girl|boy|child|kid)s?", r"girl child", r"boy child", r"loli(ta|con)?", r"shota(con)?", r"jailbait",
    r"daughter", r"son", r"niece", r"nephew", r"student in (grade|year) \d+", r"(1st|2nd|3rd|[4-9]th|1[0-2]th) grader?",
    r"barely legal", r"young[- ]looking", r"childlike", r"petite (girl|child)", r"flat[- ]chested (girl|child)",
    r"menor(es)?", r"niñ[oa]s?", r"adolescente", r"enfants?", r"fillette", r"mädchen", r"kinder?", r"jugendliche",
    "儿童", "幼女", "幼童", "少女", "少年", "未成年", "小学生", "中学生", "高中生", "萝莉", "正太", "ロリ", "ショタ", "子供", "児童",
    "女子高生", "女子中学生", "小学", "中学",
]
# ages: "15 year old", "15yo", "aged 12", "age 9", "15-year-old", "15 y/o", "15岁", "15歳"
AGE = re.compile(r"(?<!\d)(\d{1,2})\s*(-|\s)?\s*(years?|yrs?|y/?o|yo)\b|\b(aged?|age of)\s*(\d{1,2})\b|(\d{1,2})\s*(岁|歲|歳|才)", re.I)

def _norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[_*~`|]+", " ", t)
    # undo simple obfuscation: l0li, t33n, s3xy
    t = t.translate(str.maketrans({"0": "o", "3": "e", "1": "i", "4": "a", "@": "a", "$": "s"})) if re.search(r"[a-z][0134@$][a-z]", t) else t
    return t

def _rx(words):
    lat = [w for w in words if re.match(r"^[\x00-\x7f]", w) or "\\" in w or "(" in w]
    cjk = [w for w in words if w not in lat]
    parts = [r"\b(?:" + "|".join(lat) + r")\b"] if lat else []
    if cjk:
        parts.append("(?:" + "|".join(re.escape(w) for w in cjk) + ")")
    return re.compile("|".join(parts), re.I)

SEX_RX, MINOR_RX = _rx(SEXUAL), _rx(MINOR)

NUMW = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                    "sixteen seventeen".split())}
AGEW = re.compile(r"\b(" + "|".join(NUMW) + r")[\s-]*(years?|yrs?|y/?o)\b", re.I)

def minor_age(t):
    m = AGEW.search(t)
    if m:
        return NUMW[m.group(1).lower()]
    for m in AGE.finditer(t):
        n = next((g for g in (m.group(1), m.group(5), m.group(6)) if g), None)
        if n is not None and int(n) < 18:
            return int(n)
    return None

def check_prompt(text):
    t = _norm(text or "")
    t_raw = unicodedata.normalize("NFKC", text or "").lower()
    sex = SEX_RX.search(t) or SEX_RX.search(t_raw)
    if not sex:
        return True, "ok"
    mi = MINOR_RX.search(t) or MINOR_RX.search(t_raw)
    age = minor_age(t_raw)
    if mi or age is not None:
        return False, f"blocked: sexual/nudity term '{sex.group(0)}' with minor indicator '{mi.group(0) if mi else str(age) + ' years'}'"
    return True, "ok (adult content allowed)"

GUARD_ID = "minors-v1"
REFUSAL = "Prompt refused: sexual content involving minors is not allowed"
# canaries for the fail-closed load check (text only; the engine refuses to start if any canary is misclassified)
_MUST_BLOCK = ("nude 15 year old", "sexy schoolgirl", "naked child", "nsfw loli", "s3xy t33n", "裸体 萝莉")
_MUST_ALLOW = ("nude adult woman, 30 years old, fine art photography", "a child flying a kite in a park",
               "a red vintage bicycle with a wicker basket")

def self_test():
    """Raise RuntimeError unless the guard classifies every canary correctly; returns the guard id."""
    wrong = [p for p in _MUST_BLOCK if check_prompt(p)[0]] + [p for p in _MUST_ALLOW if not check_prompt(p)[0]]
    if wrong:
        raise RuntimeError(f"prompt guard self-test failed on {len(wrong)} canaries; refusing to serve")
    return GUARD_ID

def check_prompts(prompt, prompt_embeds=None):
    """Pipeline-level gate: raise ValueError for a refused prompt (str or list) and for precomputed prompt embeddings
    without their text, which would bypass the text check."""
    if prompt is None:
        if prompt_embeds is not None:
            raise ValueError("Prompt refused: precomputed prompt embeddings bypass the prompt guard")
        return
    for p in [prompt] if isinstance(prompt, str) else list(prompt):
        if not isinstance(p, str) or not check_prompt(p)[0]:
            raise ValueError(REFUSAL)

def install(pipeline):
    """Wrap pipeline.encode_prompt so every prompt encoding (positive and negative, any caller) passes the guard first.
    Returns the wrapper; the engine checks before each request that it is still the pipeline's encode_prompt."""
    original = pipeline.encode_prompt
    def encode_prompt(*args, **kwargs):
        check_prompts(kwargs.get("prompt", args[0] if args else None), kwargs.get("prompt_embeds"))
        return original(*args, **kwargs)
    encode_prompt.paiton_prompt_guard = GUARD_ID
    pipeline.encode_prompt = encode_prompt
    return encode_prompt
