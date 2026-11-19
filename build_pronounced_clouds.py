import pymupdf
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

# 1. Load PDF Page 3 clean assets (im151 = Celestia background, im160 = top sky vignette)
doc = pymupdf.open('visual arts 1-3.pdf')

def get_pixmap_rgba(doc, xref):
    smask = doc.xref_get_key(xref, 'SMask')
    pix = pymupdf.Pixmap(doc, xref)
    if smask[0] != 'null':
        smask_xref = int(smask[1].split()[0])
        smask_pix = pymupdf.Pixmap(doc, smask_xref)
        pix_rgba = pymupdf.Pixmap(pix, smask_pix)
        return Image.frombytes('RGBA', [pix_rgba.width, pix_rgba.height], pix_rgba.samples)
    elif pix.alpha:
        return Image.frombytes('RGBA', [pix.width, pix.height], pix.samples)
    else:
        return Image.frombytes('RGB', [pix.width, pix.height], pix.samples).convert('RGBA')

w_target, h_target = 2048, 1320
scale = 2048 / 1024.5

# Composite clean upper scene from PDF
clean_canvas = Image.new('RGBA', (w_target, h_target), (255, 230, 205, 255))

# im151: Celestia sky, pillars, doorway
im151 = get_pixmap_rgba(doc, 151)
bg_w = int((1062.46 - (-37.788)) * scale)
bg_h = int(618.75 * scale)
bg_x = int(-37.788 * scale)
bg_r = im151.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
clean_canvas.paste(bg_r, (bg_x, 0))

# im160: Sky top vignette
im160 = get_pixmap_rgba(doc, 160)
im160_w = int((1025.43 - (-0.657)) * scale)
im160_h = int(309.40 * scale)
im160_r = im160.resize((im160_w, im160_h), Image.Resampling.LANCZOS)
clean_canvas.alpha_composite(im160_r, (int(-0.657 * scale), 0))

# 2. Load ref2 (media_1795093204273_39668b17.jpg) for authentic pronounced clouds & walkway
ref2 = Image.open('C:/Users/Unbidden/.gemini/antigravity/brain/296ab444-449c-43cd-a6b8-9a94f8d6a1c0/.user_uploaded/media_1795093204273_39668b17.jpg')
ref2_2x = ref2.resize((w_target, h_target), Image.Resampling.LANCZOS)

# Inpaint chevron artifact in ref2_2x around y=940..1015, x=990..1055 using surrounding symmetric walkway paving
arr_ref2 = np.array(ref2_2x).copy()
for x in range(992, 1056):
    t = (x - 992) / (1056 - 992)
    x_l = max(0, 992 - (x - 992))
    x_r = min(w_target - 1, 1056 + (1056 - x))
    arr_ref2[940:1015, x] = (arr_ref2[940:1015, x_l].astype(float) * (1.0 - t) + arr_ref2[940:1015, x_r].astype(float) * t).astype(np.uint8)

# 3. Construct smooth blend mask
# Above y=680 in center: 100% clean canvas (NO logo, NO tagline)
# Below y=750 in center: 100% ref2_2x (pronounced clouds, walkway)
# On sides (x < 420 or x > 1628): clouds start around y=560..640
mask = np.zeros((h_target, w_target), dtype=float)

for y in range(h_target):
    for x in range(w_target):
        dist = abs(x - 1024)
        if dist < 600:
            y_start, y_end = 680, 750
        else:
            y_start, y_end = 560, 640
            
        if y <= y_start:
            mask[y, x] = 0.0
        elif y >= y_end:
            mask[y, x] = 1.0
        else:
            # Smooth cosine s-curve
            t = (y - y_start) / (y_end - y_start)
            mask[y, x] = 0.5 - 0.5 * np.cos(t * np.pi)

mask_smooth = gaussian_filter(mask, sigma=12)

# Blend
arr_clean = np.array(clean_canvas)[:, :, :3].astype(float)
arr_ref = arr_ref2.astype(float)

mask_3d = mask_smooth[:, :, np.newaxis]
blended = arr_clean * (1.0 - mask_3d) + arr_ref * mask_3d

# Ensure bottom 50px fade smoothly and strictly into #ffe6cd (255, 230, 205)
PEACH = np.array([255.0, 230.0, 205.0])
fade_start = h_target - 50
for y in range(fade_start, h_target):
    t = (y - fade_start) / 50.0
    alpha = t * t * (3.0 - 2.0 * t) # smoothstep
    blended[y] = blended[y] * (1.0 - alpha) + PEACH * alpha

# Strictly lock bottom 25 rows to pure solid PEACH
blended[-25:, :] = PEACH

final_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

# Save both lossless WebP and PNG
final_img.save('assets/images/hero_perfect_bg.webp', lossless=True)
final_img.save('assets/images/hero_perfect_bg.png')

print('Successfully generated assets/images/hero_perfect_bg.webp and .png!')
print('Bottom row colors unique:', np.unique(np.array(final_img)[-1, :, :], axis=0))
print('Size:', final_img.size)
