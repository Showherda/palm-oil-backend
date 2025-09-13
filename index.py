# api/index.py
# FastAPI backend for palm oil reconnaissance forms with image processing
# Handles timestamp-based image mapping, blob storage, and PostgreSQL database operations

import os, io, time, hashlib, logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, Request, HTTPException
from PIL import Image
from PIL.ExifTags import TAGS
import asyncpg
import asyncio
import httpx
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Palm Oil Reconnaissance API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security configuration
ALLOWED_BLOB_DOMAINS = [
    "blob.vercel-storage.com",
    ".public.blob.vercel-storage.com"  # Allow any subdomain
]

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

def validate_blob_url(url: str) -> bool:
    """Validate that URL is from allowed blob storage domains (SSRF protection)"""
    try:
        parsed = urlparse(url)
        
        # Must be HTTPS
        if parsed.scheme != "https":
            logger.warning(f"Non-HTTPS URL rejected: {url}")
            return False
        
        # Must be from allowed domains
        hostname = parsed.hostname
        if not hostname:
            logger.warning(f"No hostname in URL: {url}")
            return False
        
        # Check against allowed domains
        for allowed_domain in ALLOWED_BLOB_DOMAINS:
            if allowed_domain.startswith("."):
                # Subdomain match (e.g., .public.blob.vercel-storage.com)
                if hostname.endswith(allowed_domain[1:]):
                    logger.debug(f"URL validated: {url}")
                    return True
            else:
                # Exact domain match
                if hostname == allowed_domain:
                    logger.debug(f"URL validated: {url}")
                    return True
        
        logger.warning(f"URL from disallowed domain rejected: {url}")
        return False
        
    except Exception as e:
        logger.error(f"URL validation error for {url}: {e}")
        return False

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for API key authentication and request validation"""
    
    # Skip security for non-API endpoints
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    
    # API Key authentication
    api_key = request.headers.get("X-API-Key")
    expected_key = os.environ.get("API_KEY")
    
    if not expected_key:
        logger.error("API_KEY environment variable not set")
        raise HTTPException(500, "Server configuration error")
    
    if api_key != expected_key:
        logger.warning(f"Invalid API key from {request.client.host}")
        raise HTTPException(401, "Invalid API key")
    
    # No request size limit - allow large batches of images
    
    return await call_next(request)

async def download_image_from_url(url: str) -> Optional[bytes]:
    """Download image from URL and return bytes with security validation"""
    try:
        # Validate URL to prevent SSRF
        if not validate_blob_url(url):
            logger.warning(f"URL validation failed: {url}")
            return None
        
        logger.debug(f"Downloading image from URL: {url}")
        
        # Set timeouts and limits for security
        timeout = httpx.Timeout(30.0)  # 30 second timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            img_bytes = response.content
            logger.info(f"Downloaded image: {len(img_bytes)} bytes from {url}")
            return img_bytes
            
    except httpx.TimeoutException:
        logger.error(f"Timeout downloading image from {url}")
        return None
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None

def extract_image_timestamp(img_bytes: bytes) -> Optional[int]:
    """Extract timestamp from image EXIF data (DateTimeOriginal tag)
    
    Returns timestamp in milliseconds or None if not found
    """
    try:
        # Open image with PIL
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        # Get EXIF data
        exif_data = pil_image.getexif()
        if not exif_data:
            logger.debug("No EXIF data found in image")
            return None
        
        # Look for DateTimeOriginal tag (tag 36867)
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "DateTimeOriginal":
                logger.debug(f"Found DateTimeOriginal: {value}")
                
                # Parse the datetime string (format: "YYYY:MM:DD HH:MM:SS")
                dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                timestamp_ms = int(dt.timestamp() * 1000)
                logger.info(f"Extracted timestamp from EXIF: {timestamp_ms} ({value})")
                return timestamp_ms
        
        logger.debug("DateTimeOriginal tag not found in EXIF data")
        return None
        
    except Exception as e:
        logger.error(f"Failed to extract timestamp from EXIF: {e}")
        return None



async def find_form_by_timestamp(image_timestamp: int) -> Optional[int]:
    """Find the most recent form created before the given image timestamp"""
    logger.debug(f"Looking for form created before timestamp: {image_timestamp}")
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            "SELECT id, tree_id FROM recon_forms WHERE created_at <= $1 ORDER BY created_at DESC LIMIT 1", 
            image_timestamp
        )
        
        if r:
            form_id = r['id']
            tree_id = r['tree_id']
            logger.info(f"Found form {form_id} (tree: {tree_id}) for timestamp {image_timestamp}")
            return form_id
        else:
            logger.debug(f"No forms found before timestamp {image_timestamp}")
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
@limiter.limit("20/minute")  # 20 form creations per minute
async def create_form(request: Request):
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
        body = await request.json()
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
        raise HTTPException(500, "Internal server error")

@app.post("/api/image-list")
@limiter.limit("10/minute")  # 10 image batch uploads per minute
async def process_blob_images(request: Request):
    """Process a list of images already uploaded to blob storage
    
    Images are downloaded from their blob URLs and processed with timestamp-based form mapping.
    Each image is associated with the most recent form created before the image timestamp.
    If no form exists before the image timestamp, the image is stored as UNMATCHED.
    
    Expected payload:
    {
        "images": [
            {
                "url": "https://blob-url.com/path/to/image.jpg",
                "filename": "optional filename",
                "timestamp": optional_fallback_timestamp_ms
            }
        ]
    }
    """
    logger.info("Processing blob image list")
    
    try:
        body = await request.json()
        images = body.get("images", [])
        
        if not isinstance(images, list) or len(images) == 0:
            logger.error("Invalid images list provided")
            raise HTTPException(400, "images list required")
        
        # Allow unlimited number of images per request
        
        logger.info(f"Processing {len(images)} blob images")
        
        processed = 0
        errors: List[Dict[str,Any]] = []
        
        for i, item in enumerate(images):
            try:
                logger.debug(f"Processing blob image {i+1}/{len(images)}")
                
                image_url = item.get("url")
                filename = item.get("filename") or f"image_{i}.jpg"
                fallback_timestamp = item.get("timestamp", now_ms())
                
                if not image_url:
                    logger.warning(f"Image {i} has no URL")
                    errors.append({"index": i, "reason": "no URL provided"})
                    continue
                
                # Download image from blob storage
                img_bytes = await download_image_from_url(image_url)
                if not img_bytes:
                    logger.error(f"Failed to download image {i} from {image_url}")
                    errors.append({"index": i, "reason": "failed to download from URL", "url": image_url})
                    continue
                
                logger.debug(f"Downloaded image {i}: {len(img_bytes)} bytes")
                
                # Extract timestamp from EXIF data
                image_timestamp = extract_image_timestamp(img_bytes)
                if image_timestamp is None:
                    logger.debug(f"No EXIF timestamp found for image {i}, using fallback: {fallback_timestamp}")
                    image_timestamp = fallback_timestamp
                
                logger.info(f"Blob image {i} timestamp: {image_timestamp}")
                
                # Find form based on timestamp
                form_id = await find_form_by_timestamp(image_timestamp)
                
                if not form_id:
                    logger.info(f"Blob image {i} has no matching form (timestamp: {image_timestamp}), marking as UNMATCHED")
                    
                    # Find or create UNMATCHED form
                    unmatched_form = await find_form_by_timestamp(now_ms())
                    if not unmatched_form:
                        unmatched_form = await create_placeholder("UNMATCHED")
                    
                    checksum = sha256(img_bytes)
                    await append_image(unmatched_form, image_url, filename, checksum)
                    errors.append({
                        "index": i, 
                        "reason": "no form found for timestamp; stored under UNMATCHED", 
                        "url": image_url, 
                        "timestamp": image_timestamp
                    })
                    processed += 1
                    continue
                
                # Get form details
                pool = await get_pool()
                async with pool.acquire() as conn:
                    form_data = await conn.fetchrow("SELECT tree_id FROM recon_forms WHERE id=$1", form_id)
                    tree_id = form_data['tree_id'] if form_data else 'UNKNOWN'
                
                logger.info(f"Mapping blob image {i} to form {form_id} (tree: {tree_id})")
                
                checksum = sha256(img_bytes)
                ok = await append_image(form_id, image_url, filename, checksum)
                
                if ok:
                    processed += 1
                    logger.info(f"Successfully processed blob image {i}: {image_url} -> form {form_id}")
                else:
                    logger.warning(f"Duplicate blob image detected at index {i}")
                    errors.append({"index": i, "reason": "duplicate checksum; skipped"})
                    
            except Exception as e:
                logger.error(f"Error processing blob image {i}: {e}")
                errors.append({"index": i, "error": str(e)})
        
        logger.info(f"Blob image processing complete: {processed} processed, {len(errors)} errors")
        return {"processed": processed, "errors": errors}
        
    except Exception as e:
        logger.error(f"Blob image list processing failed: {e}")
        raise HTTPException(500, "Internal server error")
