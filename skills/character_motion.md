# Character Motion Skill (角色动作编排)

Purpose: describe what the character's BODY does across the shot as a time-sequenced
choreography, so AI video models render lively, readable motion instead of a moving photo.
This is about MOTION over time, not a richer pose. Applies to every clip, not just action.

## Decompose into time-weighted micro-beats
- Break the shot into 1–3 concrete physical beats (a short clip renders 1–3 movements well —
  see the density ceiling). Each beat is ONE clear body action, not a vague label.
- Weight the beats by how long each earns on screen: a fast strike is brief, a held reaction
  lingers. Do not split time evenly by default; let the drama decide.

## Every movement: anticipation → action → contact → follow-through
- Anticipation: the load/wind-up before the move (weight shifts to the back foot, blade drawn back).
- Action: the movement itself, along a clear arc, with real weight and momentum.
- Contact/consequence: where force lands and what reacts (dust kicked up, a guard buckles, water sprayed).
- Follow-through & overlapping motion: the body settles a beat later; cloth, hair, and breath LAG
  behind the main motion and keep moving after it stops. Name this secondary motion explicitly.

## Convey speed through RELATIVE motion, not a fast camera
- To read as fast: the environment streaks past with heavy motion blur, limbs blur at the extremes,
  cloth snaps taut, and the camera can move OPPOSITE the subject so they scissor past each other.
- Do NOT ask for a fast camera — that warps the model. A calm camera + fast subject reads as fast.

## Density ceiling (important)
- A ~5s clip reliably shows 1–3 distinct movements. Describe FEWER movements in RICHER physical
  detail rather than cramming many — crammed movements blur into mush.
- A dense exchange (a full fight) is a SEQUENCE of clips, one exchange per clip — not one clip.

## Ground the body
- State weight distribution, balance, which foot/hand leads, joints bending, and contact with the
  ground or props. Motion without weight and contact floats.

---

# 角色动作编排技能（中文）

目的：把角色身体在镜头内随时间发生的动作，描述成一段有先后、有节奏的编排，让 AI 视频模型生成
生动、可读的动作，而不是一张会动的照片。重点是"随时间变化的动作"，不是更详细的静态姿势。适用于每一个镜头，不只是动作戏。

## 拆解为带时长权重的微动作
- 将镜头拆成 1–3 个具体的身体动作（短镜头只能清晰呈现 1–3 个动作，见密度上限）。每个动作是一个清晰的身体动作，不是笼统的标签。
- 按各动作应占的时长加权：快速的出招很短，停顿的反应更久。默认不要平均分配时间，让戏剧节奏决定。

## 每个动作：预备 → 动作 → 触碰 → 随势收势
- 预备：出手前的蓄力（重心移到后脚、剑向后拉）。
- 动作：沿清晰弧线完成，带真实的重量与惯性。
- 触碰/后果：力落在哪里、什么被带动（扬起尘土、格挡被压垮、水花溅起）。
- 随势与连带运动：身体略迟一拍才收住；衣料、头发、呼吸滞后于主体动作，并在主体停下后继续运动。明确写出这些连带运动。

## 用相对运动表现速度，而非快速运镜
- 表现"快"：背景带强烈运动模糊快速掠过、四肢在极点处产生拖影、衣料被气流绷紧，镜头可与主体反向运动形成交错。
- 不要要求快速运镜——那会让模型变形。静稳的镜头 + 快速的主体，看起来就是快。

## 密度上限（重要）
- 约 5 秒的镜头只能清晰呈现 1–3 个动作。宁可把更少的动作描述得更细，也不要塞满——塞满会糊成一团。
- 密集的交手（一整场打斗）是一串镜头，一次交手一个镜头，而不是一个镜头。

## 让身体有根
- 写明重心分布、平衡、哪只手/脚在前、关节如何弯曲、与地面或道具的接触。没有重量与接触的动作会飘。
