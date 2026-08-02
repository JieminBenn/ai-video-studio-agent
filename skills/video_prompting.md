# Video Prompting Skill

Purpose: turn a structured video brief + plain user intent into ONE dense, model-ready
prompt that drives clear, intentional MOTION for image-to-video models (Veo, Seedance,
Kling, Grok Imagine Video). The user often cannot prompt well — do the craft for them.

## Write like this
- Lead with the SUBJECT and the single action the shot must sell, then describe how it
  moves over time. Motion first, not a static description.
- Keep it concrete: name the physical change, weight, follow-through, and the final pose
  the editor can cut on. Specify camera move, lens feel, lighting, palette, and mood.
- Expand each motion beat into a real choreography — anticipation, the action along its arc,
  contact/consequence, and follow-through with overlapping secondary motion (cloth/hair/breath
  lag). Follow the Character motion skill; convey speed through relative motion and blur, not a
  faster camera. Keep the brief's beat order and timing.
- Density over scaffolding: no markdown headings, no bullet dumps, no skill text.
- Aim for ~60–130 words.
- **Ban empty hype words** ("cinematic", "epic", "stunning", "masterpiece", "high quality",
  "4k") — they don't render. Replace each with the concrete motion, weight, light, or lens
  that would earn it.

## Per-model notes (adapt to the target video model)
- **Seedance**: excels at a single continuous shot with strong subject motion; keep to ONE
  camera move and one clear action arc, and preserve the brief's colon-form dialogue for its
  native speech.
- **Kling**: best native dialogue + lip-sync; when there are lines, foreground clear mouth
  timing and keep the action from fighting the speaking performance.
- **Veo**: dialogue plus synchronized SFX and commercial polish; state the diegetic sound
  intent so audio and picture lock together.
- Whatever the model, keep motion motivated and continuous — the concrete beats travel across
  all of them; do not rely on model-specific token tricks.

## PRESERVE from the brief (these are constraints, not suggestions)
- The time-keyed **motion beats** and their ordering/timing — every second must move.
- The exact **camera move** (and any camera-movement recipe language) — execute it precisely.
- **Continuity / last-frame** instructions: if the brief says begin on the previous shot's
  final frame, keep that; otherwise do not invent continuity the provider can't do.
- The **exact spoken dialogue** lines, verbatim and in order — never add, drop, paraphrase, or
  reassign words. Keep them in the brief's **colon form** (`Name: line`); do NOT wrap them in quotes.
  Preserve the intelligibility direction with them: clearly enunciated and fully intelligible in the
  stated language, visible mouth movement, a natural pace that fits the shot length, and
  **no on-screen text / captions / subtitles**. If nobody speaks, keep it silent.
- **No background music** — keep the brief's no-score line; the model must not add music (the score,
  if any, is a separate muxed stem).
- **Sound** intent (diegetic SFX, ambience, no music) and any **provider capability limits**
  the brief states (e.g. no extra reference media) — do not contradict them.
- Identity references: faces, wardrobe, and proportions must match the supplied references.

## Style and intent
- Lean into the named style's playbook idiom (render look, materials, lighting, palette).
- Translate vibe words into craft, tastefully and within provider safety: "sexy" →
  confident motion, glamour light, elegant wardrobe; "realistic" → natural skin/physics,
  believable weight and momentum. Never explicit content.

## Negatives
Short comma list of motion/anatomy defects to avoid: static or frozen frame, slideshow,
morphing, warping, flicker, duplicated limbs, deformed hands, jittery motion, identity
drift, lip-sync drift, plus style-specific failure modes.

## Output
Return only JSON: {"prompt": "...", "negative": "..."}. No markdown around it.
