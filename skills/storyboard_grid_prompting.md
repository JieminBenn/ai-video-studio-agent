# Storyboard grid (分镜图) prompting

You are compiling ONE image prompt that renders a single composite **storyboard contact
sheet**: a strict R×C grid of equal panels that a Seedance video model will read as a shot
sequence. Quality here means the grid is clean enough to follow and consistent enough to
read as one character in one place.

## Non-negotiable structure
- State the exact panel count and the exact `R rows × C columns` layout, read
  left-to-right, top-to-bottom.
- Uniform panel size; thin, clean gutters; each panel a 16:9 frame.
- No panel numbers, captions, text, or decorative borders — they corrupt the read.

## Consistency (the whole point of one image)
- Same character(s), same wardrobe, hair, face, and palette in every panel.
- Same location and lighting state; only pose / action / camera advance.

## Motion progression
- Panel 1 establishes; the middle panels develop with visible physical change; the last
  panel resolves to a clean final pose. The panels must read as one continuous motion.

## Avoid
collage / scrapbook / poster layouts, uneven or irregular grids, character drift between
panels, captions or watermarks, and any single non-grid image.

（中文同样适用：九宫格＝3×3，十二宫格＝3×4；每格保持同一角色、同一场景，仅动作与镜头推进。）
