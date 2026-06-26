# filters.py
import cv2
import numpy as np

def _apply_matrix(frame, matrix):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    filtered = np.clip(rgb.reshape((-1, 3)).dot(matrix.T), 0, 255).astype(np.uint8)
    return cv2.cvtColor(filtered.reshape(frame.shape), cv2.COLOR_RGB2BGR)

# 1 - 5
def apply_dog_vision(frame):
    return _apply_matrix(frame, np.array([[0.20, 0.99, -0.19], [0.16, 0.79, 0.04], [0.01, -0.01, 1.00]]))

def apply_cat_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*0.35, 0, 255).astype(np.uint8)
    low_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    r, c = frame.shape[:2]
    m = np.zeros((r, c), dtype=np.float32)
    cv2.circle(m, (c//2, r//2), int(min(r,c)*0.35), 1, -1)
    m = np.dstack([cv2.GaussianBlur(m, (99,99), 0)]*3)
    return (low_sat*m + cv2.GaussianBlur(low_sat, (35,35), 0)*(1-m)).astype(np.uint8)

def apply_bee_vision(frame):
    r, c = frame.shape[:2]
    p = 12
    pix = cv2.resize(cv2.resize(frame, (c//p, r//p), interpolation=cv2.INTER_LINEAR), (c, r), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = np.mod(hsv[:,:,0]+45, 180).astype(np.uint8)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*1.8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_snake_vision(frame):
    return cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

def apply_shark_vision(frame):
    return cv2.cvtColor(cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

# 6 - 10
def apply_cuttlefish_vision(frame):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7,7), 0)
    return cv2.cvtColor(np.clip(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).reshape((-1,3)).dot(np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]).T), 0, 255).astype(np.uint8).reshape(frame.shape), cv2.COLOR_RGB2BGR)

def apply_eagle_vision(frame):
    r, c = frame.shape[:2]
    w, h = int(c*0.45), int(r*0.45)
    zoomed = cv2.resize(frame[r//2-h//2:r//2+h//2, c//2-w//2:c//2+w//2], (c, r), interpolation=cv2.INTER_CUBIC)
    cv2.circle(zoomed, (c//2, r//2), 40, (0, 255, 255), 1)
    return zoomed

def apply_owl_vision(frame):
    table = np.array([((i / 255.0) ** (1.0 / 2.5)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    hsv = cv2.cvtColor(cv2.LUT(frame, table), cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*0.7, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_frog_vision(frame):
    edges = cv2.cvtColor(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(cv2.addWeighted(cv2.GaussianBlur(frame, (25,25), 0), 0.4, frame, 0.1, 0), 0.6, edges, 0.4, 0)

def apply_horse_vision(frame):
    r, c = frame.shape[:2]
    out = frame.copy()
    cv2.rectangle(out, (c//2 - 35, 0), (c//2 + 35, r), (10, 10, 10), -1)
    return cv2.GaussianBlur(out, (5,5), 0)

# 11 - 15
def apply_pigeon_vision(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l), np.clip(a*1.5, 0, 255).astype(np.uint8), np.clip(b*1.5, 0, 255).astype(np.uint8)]), cv2.COLOR_LAB2BGR)

def apply_bat_vision(frame):
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 30, 100)
    neon = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    neon[edges > 0] = [255, 255, 0]
    return neon

def apply_rat_vision(frame):
    return cv2.GaussianBlur(frame, (45, 45), 0)

def apply_chameleon_vision(frame):
    r, c = frame.shape[:2]
    return np.hstack((cv2.flip(frame[:, :c//2], 1), frame[:, c//2:]))

def apply_shrimp_vision(frame):
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    h = np.mod(h * 3 + 60, 180).astype(np.uint8)
    s = np.clip(s * 2.5, 0, 255).astype(np.uint8)
    v = (np.sin(v / 255.0 * np.pi) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

# 16 - 20
def apply_deer_vision(frame):
    return _apply_matrix(frame, np.array([[0.43, 0.72, -0.15], [0.34, 0.65, 0.01], [-0.02, 0.03, 1.00]]))

def apply_rabbit_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*0.6, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
    return np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def apply_squid_vision(frame):
    return cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

def apply_crocodile_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = 60
    return cv2.addWeighted(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.5, frame, 0.5, 0)

def apply_butterfly_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = np.mod(hsv[:,:,0] + 130, 180).astype(np.uint8)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 2.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 21 - 25
def apply_penguin_vision(frame):
    b, g, r = cv2.split(frame)
    return cv2.merge([np.clip(b*1.5, 0, 255).astype(np.uint8), np.clip(g*1.2, 0, 255).astype(np.uint8), np.clip(r*0.4, 0, 255).astype(np.uint8)])

def apply_cow_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = np.mod(hsv[:,:,0] // 4 + 10, 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_goat_vision(frame):
    r, c = frame.shape[:2]
    return cv2.resize(cv2.resize(frame, (c, int(r * 0.75))), (c, r))

def apply_goldfish_vision(frame):
    r, c = frame.shape[:2]
    mx, my = np.meshgrid(np.arange(c), np.arange(r))
    dx, dy = mx - c/2, my - r/2
    rmat = np.sqrt(dx**2 + dy**2)
    rmat_max = np.max(rmat) if np.max(rmat) > 0 else 1
    m_x = c/2 + dx * (rmat / rmat_max) * 1.2
    m_y = r/2 + dy * (rmat / rmat_max) * 1.2
    hsv = cv2.cvtColor(cv2.remap(frame, m_x.astype(np.float32), m_y.astype(np.float32), cv2.INTER_LINEAR), cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = np.mod(hsv[:,:,0] + 75, 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_spider_vision(frame):
    return cv2.addWeighted(frame, 0.4, cv2.GaussianBlur(frame, (15,15), 0), 0.6, 0)

# --- NEW FILTERS (26 - 50) ---
def apply_reindeer_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    salju = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    gelap = cv2.bitwise_not(salju)
    return cv2.addWeighted(salju, 1.0, cv2.multiply(frame, np.array([0.2,0.2,0.2,1.0])), 0.5, 0)

def apply_mosquito_vision(frame):
    thermal = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)
    return cv2.GaussianBlur(thermal, (35,35), 0)

def apply_archerfish_vision(frame):
    r, c = frame.shape[:2]
    mx, my = np.meshgrid(np.arange(c), np.arange(r))
    m_x = (mx + 8 * np.sin(my / 10.0)).astype(np.float32)
    m_y = my.astype(np.float32)
    distorted = cv2.remap(frame, m_x, m_y, cv2.INTER_LINEAR)
    m = np.zeros((r, c), dtype=np.float32)
    cv2.circle(m, (c//2, r//2), 120, 1, -1)
    m = np.dstack([m]*3)
    return (frame * m + distorted * (1 - m)).astype(np.uint8)

def apply_dragonfly_vision(frame):
    r, c = frame.shape[:2]
    mx, my = np.meshgrid(np.arange(c), np.arange(r))
    dx, dy = mx - c/2, my - r/2
    rmat = np.sqrt(dx**2 + dy**2)
    rmat_max = np.max(rmat) if np.max(rmat) > 0 else 1
    m_x = c/2 + dx * (rmat / rmat_max) * 1.5
    m_y = r/2 + dy * (rmat / rmat_max) * 1.5
    return cv2.remap(frame, m_x.astype(np.float32), m_y.astype(np.float32), cv2.INTER_LINEAR)

def apply_dragonfish_vision(frame):
    r, c = frame.shape[:2]
    mask = np.zeros((r, c), dtype=np.uint8)
    cv2.circle(mask, (c//2, r//2), 90, 255, -1)
    red_lens = frame.copy()
    red_lens[:,:,0] = 0; red_lens[:,:,1] = 0 # Sisakan kanal Merah (BGR)
    dark = np.zeros(frame.shape, dtype=np.uint8)
    mask_3d = np.dstack([mask]*3) / 255.0
    return (red_lens * mask_3d + dark * (1 - mask_3d)).astype(np.uint8)

def apply_squirrel_vision(frame):
    return _apply_matrix(frame, np.array([[0.35, 0.65, 0.0], [0.30, 0.70, 0.0], [0.0, 0.20, 0.80]]))

def apply_chicken_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 2.2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_elephant_vision(frame):
    r, c = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (11,11), 0)
    return cv2.resize(cv2.resize(blur, (c, int(r*0.9))), (c, r))

def apply_lion_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.5, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[:,:,2] = np.clip(bgr[:,:,2]*0.6, 0, 255).astype(np.uint8) # Kurangi merah
    return bgr

def apply_mole_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def apply_jellyfish_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25,25), 0)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

def apply_snail_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (31,31), 0)
    r, c = frame.shape[:2]
    m = np.zeros((r, c), dtype=np.float32)
    cv2.circle(m, (c//2, r//2), int(min(r,c)*0.25), 1, -1)
    m = np.dstack([cv2.GaussianBlur(m, (55,55), 0)]*3)
    return (blur * m).astype(np.uint8)

def apply_dolphin_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    r, c = frame.shape[:2]
    grid = np.zeros(frame.shape, dtype=np.uint8)
    for i in range(0, r, 30):
        cv2.line(grid, (0, i), (c, i), (0, 180, 255), 1)
    return cv2.addWeighted(bgr, 0.8, grid, 0.2, 0)

def apply_vulture_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8)).apply(gray)
    return cv2.applyColorMap(clahe, cv2.COLORMAP_BONE)

def apply_octopus_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(bgr, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

def apply_sheep_vision(frame):
    r, c = frame.shape[:2]
    compressed = cv2.resize(frame, (c, int(r*0.6)))
    padded = cv2.copyMakeBorder(compressed, int(r*0.2), int(r*0.2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    return cv2.resize(padded, (c, r))

def apply_lemur_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = 15  # Set rona warna Amber/Oranye Malam
    hsv[:,:,1] = np.clip(hsv[:,:,1]*1.4, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_gecko_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*2.5, 0, 255).astype(np.uint8)
    hsv[:,:,2] = np.clip(hsv[:,:,2]*1.8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_scorpio_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cm = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
    return cv2.addWeighted(cm, 0.7, frame, 0.3, 0)

def apply_badger_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)

def apply_seal_vision(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = 45 # Dominasi spektrum Hijau Air
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_firefly_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cm = cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)
    return cv2.GaussianBlur(cm, (5,5), 0)

def apply_sloth_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.GaussianBlur(bgr, (15,15), 0)

def apply_human_night_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    green = np.zeros(frame.shape, dtype=np.uint8)
    green[:,:,1] = gray # Masukkan skala abu-abu hanya ke kanal HIJAU (BGR)
    return green