# api/index.py
# FastAPI backend for palm oil reconnaissance forms with image processing
# Handles QR code detection, blob storage, and PostgreSQL database operations

import os, io, base64, time, hashlib, json, logging
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, Request, HTTPException
import vercel_blob  # community wrapper; reads BLOB_READ_WRITE_TOKEN env var
from PIL import Image
import asyncpg
import asyncio
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

app = FastAPI(title="Palm Oil Reconnaissance API", version="1.0.0")

# --- PostgreSQL connection pool ---
# Global connection pool for efficient database connections
_pool = None

async def get_pool():
    """Get or create PostgreSQL connection pool"""
    global _pool
    if _pool is None:
        postgres_url = os.environ.get('POSTGRES_URL')
        if not postgres_url:
            logger.error("POSTGRES_URL environment variable not set")
            raise ValueError("Database URL not configured")
        
        logger.info("Creating PostgreSQL connection pool")
        _pool = await asyncpg.create_pool(
            postgres_url,
            min_size=1,
            max_size=10
        )
        logger.info("PostgreSQL connection pool created successfully")
    return _pool

async def init_db():
    """Initialize database tables if they don't exist"""
    logger.info("Initializing database tables")
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Create recon_forms table for storing reconnaissance form data
        logger.info("Creating recon_forms table if not exists")
        await conn.execute("""CREATE TABLE IF NOT EXISTS recon_forms (
                   id SERIAL PRIMARY KEY, tree_id TEXT NOT NULL,
                   plot_id TEXT, number_of_fruits INTEGER, harvest_days INTEGER,
                   created_at BIGINT, is_synced INTEGER DEFAULT 0, placeholder INTEGER DEFAULT 0
                )""")
        
        # Create images table for storing image metadata and URLs
        logger.info("Creating images table if not exists")
        await conn.execute("""CREATE TABLE IF NOT EXISTS images (
                   id SERIAL PRIMARY KEY, form_id INTEGER NOT NULL,
                   url TEXT, filename TEXT, checksum TEXT, uploaded_at BIGINT,
                   FOREIGN KEY(form_id) REFERENCES recon_forms(id)
                )""")
        
        logger.info("Database tables initialized successfully")

@app.on_event("startup")
async def startup():
    """Application startup event handler"""
    logger.info("Starting up FastAPI application")
    await init_db()
    logger.info("Application startup completed")

# --- Helper functions ---
def now_ms(): 
    """Get current timestamp in milliseconds"""
    return int(time.time()*1000)

def sha256(b: bytes): 
    """Calculate SHA256 hash of bytes"""
    return hashlib.sha256(b).hexdigest()

def b64_to_bytes(b64: str) -> bytes:
    """Convert base64 string to bytes, handling data URLs"""
    # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
    if b64.startswith("data:") and "," in b64:
        logger.debug("Removing data URL prefix from base64 string")
        b64 = b64.split(",",1)[1]
    return base64.b64decode(b64)

# QR code detection using OpenCV - returns decoded string or None
_qr = cv2.QRCodeDetector()

def decode_qr(img_bytes: bytes) -> Optional[str]:
    """Decode QR code from image bytes using OpenCV with PIL fallback"""
    logger.debug(f"Attempting to decode QR code from {len(img_bytes)} bytes")
    
    # Convert bytes to OpenCV image array
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if img is None:
        # Fallback to PIL if OpenCV fails to decode
        logger.debug("OpenCV decode failed, trying PIL fallback")
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            arr = np.array(pil)[:, :, ::-1]  # RGB to BGR conversion
            img = arr
        except Exception as e:
            logger.error(f"PIL fallback failed: {e}")
            return None
    
    # Try multi-decode first (handles multiple QR codes)
    try:
        ok, decoded_info, points, straight_qrcode = _qr.detectAndDecodeMulti(img)
        if ok and decoded_info:
            logger.debug(f"Multi-decode successful: {decoded_info}")
            # Return first decoded QR code (should contain treeId)
            result = decoded_info[0] if isinstance(decoded_info, (list,tuple)) else decoded_info
            logger.info(f"QR code decoded successfully: {result[:50]}...")  # Log first 50 chars
            return result
    except Exception as e:
        logger.debug(f"Multi-decode failed: {e}")
    
    # Fallback to single decode
    try:
        text, pts = _qr.detectAndDecode(img)
        if text:
            logger.info(f"Single QR decode successful: {text[:50]}...")  # Log first 50 chars
            return text
        else:
            logger.debug("No QR code detected in image")
            return None
    except Exception as e:
        logger.error(f"QR decode failed: {e}")
        return None

def parse_treeid(decoded: str) -> Optional[str]:
    """Extract tree ID from decoded QR code string
    
    Supports multiple formats:
    - JSON: {"treeId": "123"} or {"id": "123"}
    - Colon-separated: "tree:123"
    - Plain string: "123"
    """
    if not decoded:
        logger.debug("No decoded string provided")
        return None
    
    logger.debug(f"Parsing tree ID from: {decoded}")
    
    # Try to parse as JSON first
    try:
        obj = json.loads(decoded)
        if isinstance(obj, dict):
            tree_id = obj.get("treeId") or obj.get("id")
            if tree_id:
                logger.info(f"Extracted tree ID from JSON: {tree_id}")
                return tree_id
    except Exception as e:
        logger.debug(f"Not valid JSON: {e}")
    
    # Try colon-separated format
    if ":" in decoded:
        tree_id = decoded.split(":",1)[1].strip()
        logger.info(f"Extracted tree ID from colon format: {tree_id}")
        return tree_id
    
    # Use as plain string
    tree_id = decoded.strip()
    logger.info(f"Using decoded string as tree ID: {tree_id}")
    return tree_id

def upload_blob(name: str, payload: bytes) -> Optional[str]:
    """Upload file to Vercel Blob storage
    
    Args:
        name: File name/path for the blob
        payload: File content as bytes
        
    Returns:
        Public URL of uploaded blob or None if failed
    """
    logger.info(f"Uploading blob: {name} ({len(payload)} bytes)")
    
    # Ensure BLOB_READ_WRITE_TOKEN is configured
    if not os.environ.get('BLOB_READ_WRITE_TOKEN'):
        logger.error("BLOB_READ_WRITE_TOKEN environment variable not set")
        return None
    
    try:
        resp = vercel_blob.put(name, payload, {"addRandomSuffix": "true", "access": "public"})
        
        if isinstance(resp, dict):
            url = resp.get("url") or resp.get("downloadUrl")
            if url:
                logger.info(f"Blob uploaded successfully: {url}")
                return url
            else:
                logger.error(f"No URL in blob response: {resp}")
                return None
        else:
            logger.error(f"Invalid blob response type: {type(resp)}")
            return None
            
    except Exception as e:
        logger.error(f"Blob upload failed for {name}: {e}")
        return None

async def find_latest_unsynced_form(tree_id: str) -> Optional[int]:
    """Find the most recent unsynced form for a given tree ID"""
    logger.debug(f"Looking for latest unsynced form for tree: {tree_id}")
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            "SELECT id FROM recon_forms WHERE tree_id=$1 AND is_synced=0 ORDER BY created_at DESC LIMIT 1", 
            tree_id
        )
        
        if r:
            form_id = r['id']
            logger.info(f"Found unsynced form {form_id} for tree {tree_id}")
            return form_id
        else:
            logger.debug(f"No unsynced forms found for tree {tree_id}")
            return None

async def create_placeholder(tree_id: str) -> int:
    """Create a placeholder form for a tree ID to collect images"""
    logger.info(f"Creating placeholder form for tree: {tree_id}")
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        form_id = await conn.fetchval(
            "INSERT INTO recon_forms(tree_id, created_at, placeholder) VALUES ($1, $2, 1) RETURNING id", 
            tree_id, now_ms()
        )
        
        logger.info(f"Created placeholder form {form_id} for tree {tree_id}")
        return form_id

async def append_image(form_id: int, url: Optional[str], filename: Optional[str], checksum: str) -> bool:
    """Add image metadata to a form
    
    Args:
        form_id: Form to attach image to
        url: Public URL of uploaded image
        filename: Original filename
        checksum: SHA256 hash for duplicate detection
        
    Returns:
        True if image was added, False if duplicate detected
    """
    logger.debug(f"Adding image to form {form_id}: {filename} (checksum: {checksum[:16]}...)")
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Check for duplicates based on form_id and checksum
        existing = await conn.fetchrow(
            "SELECT id FROM images WHERE form_id=$1 AND checksum=$2", 
            form_id, checksum
        )
        
        if existing:
            logger.warning(f"Duplicate image detected for form {form_id}, checksum: {checksum[:16]}...")
            return False
        
        # Insert new image record
        await conn.execute(
            "INSERT INTO images(form_id,url,filename,checksum,uploaded_at) VALUES ($1,$2,$3,$4,$5)",
            form_id, url, filename, checksum, now_ms()
        )
        
        logger.info(f"Image added successfully to form {form_id}: {filename}")
        return True

# --- endpoints ---

@app.post("/api/forms")
async def create_form(req: Request):
    """Create a new reconnaissance form
    
    Expected payload:
    {
        "treeId": "string (required)",
        "plotId": "string (optional)", 
        "numberOfFruits": "integer (optional)",
        "harvestDays": "integer (optional)"
    }
    """
    logger.info("Creating new reconnaissance form")
    
    try:
        body = await req.json()
        logger.debug(f"Form creation request: {body}")
        
        tree = body.get("treeId")
        if not tree:
            logger.error("Form creation failed: treeId required")
            raise HTTPException(400, "treeId required")
        
        pool = await get_pool()
        async with pool.acquire() as conn:
            form_id = await conn.fetchval(
                "INSERT INTO recon_forms(tree_id, plot_id, number_of_fruits, harvest_days, created_at) VALUES ($1,$2,$3,$4,$5) RETURNING id",
                tree, body.get("plotId"), body.get("numberOfFruits"), body.get("harvestDays"), now_ms()
            )
            
            logger.info(f"Form created successfully: {form_id} for tree {tree}")
            return {"formId": form_id}
            
    except Exception as e:
        logger.error(f"Form creation failed: {e}")
        raise HTTPException(500, f"Form creation failed: {str(e)}")

@app.post("/api/image-list")
async def image_list(req: Request):
    """Process a list of images with QR code detection and blob storage
    
    Images with QR codes set the current tree context.
    Subsequent images without QR codes are associated with the current tree.
    Images without context are stored under 'UNMATCHED' tree.
    
    Expected payload:
    {
        "images": [
            {
                "content": "base64 encoded image data",
                "filename": "optional filename"
            }
        ]
    }
    """
    logger.info("Processing image list")
    
    try:
        body = await req.json()
        images = body.get("images", [])
        
        if not isinstance(images, list) or len(images) == 0:
            logger.error("Invalid images list provided")
            raise HTTPException(400, "images list required")
        
        logger.info(f"Processing {len(images)} images")
        
        # State tracking for current tree/form context
        current_tree = None
        current_form = None
        processed = 0
        errors: List[Dict[str,Any]] = []
        
        for i, item in enumerate(images):
            try:
                logger.debug(f"Processing image {i+1}/{len(images)}")
                
                b64 = item.get("content")
                filename = item.get("filename") or f"image_{i}.jpg"
                
                if not b64:
                    logger.warning(f"Image {i} has no content")
                    errors.append({"index": i, "reason": "no content"})
                    continue
                
                # Convert base64 to bytes
                img_bytes = b64_to_bytes(b64)
                logger.debug(f"Converted image {i} to {len(img_bytes)} bytes")
                
                # Try to decode QR code
                decoded = decode_qr(img_bytes)
                
                if decoded:
                    # QR code found - extract tree ID and set context
                    tree_id = parse_treeid(decoded)
                    if tree_id:
                        logger.info(f"QR code detected in image {i}: tree {tree_id}")
                        current_tree = tree_id
                        
                        # Find or create form for this tree
                        fid = await find_latest_unsynced_form(tree_id)
                        current_form = fid if fid else await create_placeholder(tree_id)
                        logger.info(f"Set current context: tree={current_tree}, form={current_form}")
                        continue
                    else:
                        logger.warning(f"QR code in image {i} could not be parsed: {decoded}")
                        errors.append({"index": i, "reason": "qr parsed but no treeId", "decoded": decoded})
                        continue
                
                # Non-QR image - upload to current context or UNMATCHED
                if current_tree is None or current_form is None:
                    logger.info(f"Image {i} has no tree context, storing as UNMATCHED")
                    
                    # Upload to unmatched folder
                    name = f"unmatched/{int(time.time()*1000)}_{i}.jpg"
                    url = upload_blob(name, img_bytes)
                    checksum = sha256(img_bytes)
                    
                    # Ensure UNMATCHED placeholder form exists
                    fid = await find_latest_unsynced_form("UNMATCHED")
                    if not fid:
                        fid = await create_placeholder("UNMATCHED")
                    
                    await append_image(fid, url, filename, checksum)
                    errors.append({"index": i, "reason": "no current tree; stored under UNMATCHED", "url": url})
                    processed += 1
                    continue
                
                # Upload image to current tree/form context
                logger.info(f"Uploading image {i} to tree {current_tree}, form {current_form}")
                name = f"{current_tree}/{current_form}/{int(time.time()*1000)}_{i}.jpg"
                url = upload_blob(name, img_bytes)
                
                if not url:
                    logger.error(f"Failed to upload image {i} to blob storage")
                    errors.append({"index": i, "reason": "blob upload failed"})
                    continue
                
                checksum = sha256(img_bytes)
                ok = await append_image(current_form, url, filename, checksum)
                
                if ok:
                    processed += 1
                    logger.info(f"Successfully processed image {i}: {url}")
                else:
                    logger.warning(f"Duplicate image detected at index {i}")
                    errors.append({"index": i, "reason": "duplicate checksum; skipped"})
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                errors.append({"index": i, "error": str(e)})
        
        logger.info(f"Image processing complete: {processed} processed, {len(errors)} errors")
        return {"processed": processed, "errors": errors}
        
    except Exception as e:
        logger.error(f"Image list processing failed: {e}")
        raise HTTPException(500, f"Image processing failed: {str(e)}")
