# api/index.py
import os, io, base64, time, hashlib, sqlite3, json
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, Request, HTTPException
import vercel_blob  # community wrapper; reads BLOB_READ_WRITE_TOKEN env var
from PIL import Image

app = FastAPI()

# --- in-memory DB (keeps state across warm instances) ---
_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.row_factory = sqlite3.Row
def init_db(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS recon_forms (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, tree_id TEXT NOT NULL,
                   plot_id TEXT, number_of_fruits INTEGER, harvest_days INTEGER,
                   created_at INTEGER, is_synced INTEGER DEFAULT 0, placeholder INTEGER DEFAULT 0
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS images (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, form_id INTEGER NOT NULL,
                   url TEXT, filename TEXT, checksum TEXT, uploaded_at INTEGER,
                   FOREIGN KEY(form_id) REFERENCES recon_forms(id)
                )""")
    conn.commit()
init_db(_conn)

# --- helpers ---
def now_ms(): return int(time.time()*1000)
def sha256(b: bytes): return hashlib.sha256(b).hexdigest()

def b64_to_bytes(b64: str) -> bytes:
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",",1)[1]
    return base64.b64decode(b64)

# QR detection (OpenCV). returns decoded string or None
_qr = cv2.QRCodeDetector()
def decode_qr(img_bytes: bytes) -> Optional[str]:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # try via PIL fallback
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil)[:, :, ::-1]
        img = arr
    # try multi decode first
    try:
        ok, decoded_info, points, straight_qrcode = _qr.detectAndDecodeMulti(img)
        if ok and decoded_info:
            # return first decoded (your QR scheme should put treeId in it)
            return decoded_info[0] if isinstance(decoded_info, (list,tuple)) else decoded_info
    except Exception:
        pass
    text, pts = _qr.detectAndDecode(img)
    return text if text else None

def parse_treeid(decoded: str) -> Optional[str]:
    if not decoded: return None
    try:
        obj = json.loads(decoded)
        if isinstance(obj, dict):
            return obj.get("treeId") or obj.get("id")
    except Exception:
        pass
    if ":" in decoded:
        return decoded.split(":",1)[1].strip()
    return decoded.strip()

def upload_blob(name: str, payload: bytes) -> Optional[str]:
    # uses vercel_blob.put; ensure BLOB_READ_WRITE_TOKEN env var is configured in Vercel
    try:
        resp = vercel_blob.put(name, payload, {"addRandomSuffix": "true", "access": "public"})
        return (resp.get("url") or resp.get("downloadUrl")) if isinstance(resp, dict) else None
    except Exception as e:
        print("blob upload err:", e)
        return None

def find_latest_unsynced_form(tree_id: str) -> Optional[int]:
    cur = _conn.cursor()
    cur.execute("SELECT id FROM recon_forms WHERE tree_id=? AND is_synced=0 ORDER BY created_at DESC LIMIT 1", (tree_id,))
    r = cur.fetchone()
    return r["id"] if r else None

def create_placeholder(tree_id: str) -> int:
    cur = _conn.cursor()
    cur.execute("INSERT INTO recon_forms(tree_id, created_at, placeholder) VALUES (?, ?, 1)", (tree_id, now_ms()))
    _conn.commit()
    return cur.lastrowid

def append_image(form_id: int, url: Optional[str], filename: Optional[str], checksum: str):
    cur = _conn.cursor()
    cur.execute("SELECT id FROM images WHERE form_id=? AND checksum=?", (form_id, checksum))
    if cur.fetchone(): return False
    cur.execute("INSERT INTO images(form_id,url,filename,checksum,uploaded_at) VALUES (?,?,?,?,?)",
                (form_id, url, filename, checksum, now_ms()))
    _conn.commit()
    return True

# --- endpoints ---

@app.post("/api/forms")
async def create_form(req: Request):
    body = await req.json()
    tree = body.get("treeId")
    if not tree: raise HTTPException(400, "treeId required")
    cur = _conn.cursor()
    cur.execute("INSERT INTO recon_forms(tree_id, plot_id, number_of_fruits, harvest_days, created_at) VALUES (?,?,?,?,?)",
                (tree, body.get("plotId"), body.get("numberOfFruits"), body.get("harvestDays"), now_ms()))
    _conn.commit()
    return {"formId": cur.lastrowid}

@app.post("/api/image-list")
async def image_list(req: Request):
    body = await req.json()
    images = body.get("images", [])
    if not isinstance(images, list) or len(images)==0:
        raise HTTPException(400, "images list required")
    current_tree = None
    current_form = None
    processed = 0
    errors: List[Dict[str,Any]] = []
    for i, item in enumerate(images):
        try:
            b64 = item.get("content")
            filename = item.get("filename")
            if not b64:
                errors.append({"index": i, "reason": "no content"}); continue
            img_bytes = b64_to_bytes(b64)
            decoded = decode_qr(img_bytes)
            if decoded:
                tree_id = parse_treeid(decoded)
                if tree_id:
                    current_tree = tree_id
                    fid = find_latest_unsynced_form(tree_id)
                    current_form = fid if fid else create_placeholder(tree_id)
                    continue
                else:
                    errors.append({"index": i, "reason": "qr parsed but no treeId", "decoded": decoded})
                    continue
            # non-QR
            if current_tree is None or current_form is None:
                # upload to blob under 'unmatched' for visibility
                name = f"unmatched/{int(time.time()*1000)}_{i}.jpg"
                url = upload_blob(name, img_bytes)
                checksum = sha256(img_bytes)
                # ensure there's an 'UNMATCHED' placeholder form
                fid = find_latest_unsynced_form("UNMATCHED") or create_placeholder("UNMATCHED")
                append_image(fid, url, filename, checksum)
                errors.append({"index": i, "reason": "no current tree; stored under UNMATCHED", "url": url})
                processed += 1
                continue
            # upload to blob with a name
            name = f"{current_tree}/{current_form}/{int(time.time()*1000)}_{i}.jpg"
            url = upload_blob(name, img_bytes)
            checksum = sha256(img_bytes)
            ok = append_image(current_form, url, filename, checksum)
            if ok: processed += 1
            else: errors.append({"index": i, "reason": "duplicate checksum; skipped"})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    return {"processed": processed, "errors": errors}
