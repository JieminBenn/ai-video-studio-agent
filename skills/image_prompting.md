# Image Prompting Skill

Purpose: turn a structured brief + plain user intent into ONE dense, concrete prompt that
modern image models (Gemini, gpt-image, Seedream/Doubao, Imagen) render at high quality.
The user often cannot prompt well — do the craft for them.

## Write like this
- One vivid paragraph, front-loaded with the SUBJECT, then the decisive moment/action.
- A still is ONE frozen instant — the shot's opening moment. Describe a held pose, not an
  action playing out: no camera movement, no motion blur, no sequence of events, no
  "then…" progression. If the brief narrates motion, freeze it into the single most
  telling frame. (Motion is the video prompt's job, not the keyframe's.)
- Concrete nouns and specifics over adjectives: name materials, fabrics, surfaces, props,
  hair, skin, age read, wardrobe, set dressing. "weathered brass telescope, salt-pitted"
  beats "a nice telescope".
- Always specify: shot size + lens (e.g. 85mm portrait, 35mm wide), lighting (source,
  direction, quality, contrast, time of day), color palette, mood/atmosphere, depth
  (foreground / midground / background), and a clear focal point.
- Density over instruction. Do NOT include headings, bullet lists, meta-commentary, or
  pasted reference/skill text. The model reads description, not a brief.
- Be exhaustive: describe every visible element concretely and leave nothing ambiguous — the
  model turns ambiguity into broken, illogical images. Write as richly as the input budget
  allows (you will be told a target length); every word should change the pixels.
- **Ban empty hype words.** "cinematic", "epic", "stunning", "masterpiece", "beautiful",
  "high quality", "4k", "award-winning", "breathtaking" do not render — they buy no pixels.
  Replace each with the concrete thing that would earn the word: not "cinematic lighting" but
  "low hard key from a single window, deep falloff, one rim on the jaw." Show the craft; don't
  name it.

## Per-model notes (adapt the prompt to the target model)
- **Gemini / Nano Banana** (default, reasoning + reference-grounded): natural-language
  description works best; lean on supplied reference images for identity and let precise nouns
  carry the look. Strong at multi-subject scenes when spatial relationships are stated plainly.
- **gpt-image**: strongest for legible in-image text, signage, UI, and brand/graphic layouts;
  be explicit about any text string and where it sits.
- **Seedream / Doubao (Seedream)**: responds to dense, concrete cinematographic description
  (lens, light direction, palette); keep one coherent lighting logic and avoid contradictory cues.
- Only one model runs per call — write for the concrete craft, which every model honors, rather
  than model-specific token tricks.

## Honor the references and style
- When identity references are named, state that the face, hair, wardrobe, and proportions
  must match the reference exactly; describe the person briefly but don't fight the reference.
- Lean hard into the named style's playbook idiom (render engine cues, signature materials,
  lighting and palette). A "3D" prompt should read like a polished CG feature with its own
  idiom, not a generic render.

## Aesthetic intent (tasteful, provider-safe)
- Translate vibe words into craft, not explicit content. "sexy" → confident posture,
  elegant/form-flattering wardrobe, glamour key light with soft falloff, sultry color
  grade, cinematic allure. Keep it suggestive-through-craft so providers actually render
  it; never describe nudity or explicit acts.
- "realistic face" → natural skin with pores and subsurface scattering, catchlights in the
  eyes, asymmetric real features, soft photographic depth of field, no plastic CGI sheen.
- "cinematic" → motivated lighting, controlled contrast, anamorphic/film color, deliberate
  composition.

## Negatives
Return a short comma list of defects to avoid for this image: e.g. text, watermark, logo,
extra fingers, deformed hands, fused limbs, warped anatomy, plastic skin, low detail,
blurry, collage, split panels. Add style-specific failure modes (e.g. for donghua: stiff
game-cutscene posing, Western-cartoon proportions, weightless cloth).

## Physical & logical coherence
- Ground every element: say where each subject and object sits in the frame, what supports it
  (standing on the floor, resting on the table), and what it touches or occludes.
- Keep counts and anatomy correct: the right number of people and objects, two hands, five
  fingers, connected limbs — no floating or duplicated parts.
- Hold one coherent perspective, scale, and light direction across the whole frame.
- When appearance locks are provided, reproduce them verbatim; do not paraphrase or "improve"
  the locked description.

## Output
Return only JSON: {"prompt": "...", "negative": "..."}. No markdown around it.
