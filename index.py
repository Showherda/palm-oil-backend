# api/index.py
# FastAPI backend for palm oil data collection system
# Comprehensive data collection with recon forms, harvester proofs, and tree locations
# Enhanced security, validation, and database operations

import os, io, time, hashlib, logging, re
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, Request, HTTPException, Query
from pydantic import BaseModel, validator, Field
from PIL import Image
from PIL.ExifTags import TAGS
import asyncpg
import asyncio
import httpx
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
try:
    import magic
except ImportError:
    magic = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

# Pydantic models for request validation
class ReconFormRequest(BaseModel):
    treeId: str = Field(..., min_length=1, max_length=50)
    plotId: Optional[str] = Field(None, max_length=50)
    numberOfFruits: Optional[int] = Field(None, gt=0, le=10000)
    harvestDays: Optional[int] = Field(None, ge=1, le=3)
    clientId: Optional[str] = None

    @validator('treeId')
    def validate_tree_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Tree ID is required')
        if not re.match(r'^[A-Za-z0-9_-]+$', v):
            raise ValueError('Tree ID contains invalid characters')
        return v

    @validator('plotId')
    def validate_plot_id(cls, v):
        if v:
            v = v.strip()
            if len(v) > 50:
                raise ValueError('Plot ID too long')
        return v

class ImageItem(BaseModel):
    url: str
    filename: Optional[str] = None
    timestamp: Optional[int] = None
    checksum: Optional[str] = None

class ImageListRequest(BaseModel):
    images: List[ImageItem]

class HarvesterProofRequest(BaseModel):
    treeId: str = Field(..., min_length=1, max_length=50)
    plotId: str = Field(..., min_length=1, max_length=50)
    imageUrl: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    accuracy: Optional[float] = Field(None, gt=0)
    notes: Optional[str] = Field(None, max_length=1000)
    harvesterId: Optional[str] = Field(None, max_length=100)
    clientId: Optional[str] = None

    @validator('treeId', 'plotId')
    def validate_ids(cls, v):
        v = v.strip()
        if not re.match(r'^[A-Za-z0-9_-]+$', v):
            raise ValueError('ID contains invalid characters')
        return v

class TreeLocationRequest(BaseModel):
    treeId: str = Field(..., min_length=1, max_length=50)
    plotId: str = Field(..., min_length=1, max_length=50)
    xCoordinate: float
    yCoordinate: float
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    accuracy: Optional[float] = Field(None, gt=0)
    notes: Optional[str] = Field(None, max_length=1000)
    surveyorId: Optional[str] = Field(None, max_length=100)
    clientId: Optional[str] = None

    @validator('treeId', 'plotId')
    def validate_ids(cls, v):
        v = v.strip()
        if not re.match(r'^[A-Za-z0-9_-]+$', v):
            raise ValueError('ID contains invalid characters')
        return v

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Palm Oil Data Collection API",
    version="2.0.0",
    description="Comprehensive data collection system for palm oil reconnaissance forms, harvester proofs, and tree locations"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enhanced error handling
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with detailed messages"""
    logger.warning(f"Validation error for {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": [
                {
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                }
                for error in exc.errors()
            ]
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with logging"""
    logger.error(f"Unexpected error for {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

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
    logger.info("Initializing comprehensive database tables")
    pool = await get_pool()
    async with pool.acquire() as conn:

        # Enhanced recon_forms table
        logger.info("Creating enhanced recon_forms table")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS recon_forms (
                id SERIAL PRIMARY KEY,
                tree_id VARCHAR(50) NOT NULL,
                plot_id VARCHAR(50),
                number_of_fruits INTEGER CHECK (number_of_fruits > 0),
                harvest_days INTEGER CHECK (harvest_days IN (1, 2, 3)),
                created_at BIGINT NOT NULL,
                updated_at BIGINT DEFAULT EXTRACT(EPOCH FROM NOW()) * 1000,
                is_synced BOOLEAN DEFAULT false,
                sync_timestamp BIGINT,
                client_id TEXT,
                version INTEGER DEFAULT 1,
                created_by VARCHAR(100),
                updated_by VARCHAR(100),
                placeholder INTEGER DEFAULT 0
            )
        """)

        # Enhanced images table
        logger.info("Creating enhanced images table")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                form_id INTEGER NOT NULL,
                form_type VARCHAR(20) NOT NULL DEFAULT 'recon',
                url TEXT,
                filename VARCHAR(255),
                original_filename VARCHAR(255),
                checksum VARCHAR(64) NOT NULL,
                file_size BIGINT,
                mime_type VARCHAR(50),
                image_width INTEGER,
                image_height INTEGER,
                exif_data JSONB,
                uploaded_at BIGINT NOT NULL,
                image_index INTEGER CHECK (image_index BETWEEN 1 AND 3),
                processing_status VARCHAR(20) DEFAULT 'pending'
            )
        """)

        # Harvester proofs table
        logger.info("Creating harvester_proofs table")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS harvester_proofs (
                id SERIAL PRIMARY KEY,
                tree_id VARCHAR(50) NOT NULL,
                plot_id VARCHAR(50) NOT NULL,
                image_url TEXT,
                local_image_path TEXT,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL,
                is_synced BOOLEAN DEFAULT false,
                sync_timestamp BIGINT,
                location_latitude DECIMAL(10, 8),
                location_longitude DECIMAL(11, 8),
                location_accuracy DECIMAL(6, 2),
                notes TEXT,
                harvester_id VARCHAR(100),
                client_id TEXT,
                version INTEGER DEFAULT 1
            )
        """)

        # Tree locations table
        logger.info("Creating tree_locations table")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tree_locations (
                id SERIAL PRIMARY KEY,
                tree_id VARCHAR(50) NOT NULL,
                plot_id VARCHAR(50) NOT NULL,
                x_coordinate DECIMAL(10, 6) NOT NULL,
                y_coordinate DECIMAL(10, 6) NOT NULL,
                latitude DECIMAL(10, 8),
                longitude DECIMAL(11, 8),
                gps_accuracy DECIMAL(6, 2),
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL,
                is_synced BOOLEAN DEFAULT false,
                sync_timestamp BIGINT,
                notes TEXT,
                surveyor_id VARCHAR(100),
                client_id TEXT,
                version INTEGER DEFAULT 1,
                UNIQUE (plot_id, tree_id)
            )
        """)

        # Create indexes for performance
        logger.info("Creating database indexes")

        # Recon forms indexes
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recon_forms_tree_id ON recon_forms(tree_id)")
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recon_forms_plot_id ON recon_forms(plot_id)")
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recon_forms_created_at ON recon_forms(created_at)")

        # Images indexes
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_images_form_id ON images(form_id)")
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_images_checksum ON images(checksum)")

        # Harvester proofs indexes
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_harvester_proofs_tree_id ON harvester_proofs(tree_id)")
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_harvester_proofs_plot_id ON harvester_proofs(plot_id)")

        # Tree locations indexes
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tree_locations_tree_id ON tree_locations(tree_id)")
        await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tree_locations_plot_id ON tree_locations(plot_id)")

        logger.info("Database tables and indexes initialized successfully")

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

async def validate_uploaded_file(file_content: bytes, filename: str, max_size_mb: int = 10) -> bool:
    """Validates uploaded files for security threats"""
    try:
        # Check file size
        if len(file_content) > max_size_mb * 1024 * 1024:
            raise ValueError(f"File too large (max {max_size_mb}MB)")

        # Check file type by content (not just extension)
        allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

        if magic:
            file_type = magic.from_buffer(file_content, mime=True)
            if file_type not in allowed_types:
                raise ValueError(f"Invalid file type: {file_type}")

        # Check for malicious content patterns
        if contains_malicious_content(file_content):
            raise ValueError("File contains potentially malicious content")

        # Validate image integrity with PIL
        try:
            with Image.open(io.BytesIO(file_content)) as img:
                img.verify()
        except Exception:
            raise ValueError("Invalid or corrupted image file")

        return True

    except Exception as e:
        logger.error(f"File validation failed for {filename}: {e}")
        raise

def contains_malicious_content(file_content: bytes) -> bool:
    """Scans file content for malicious patterns"""
    # Check for embedded scripts or executables
    malicious_patterns = [
        b'<script', b'javascript:', b'vbscript:',
        b'<?php', b'<%', b'exec(', b'eval(',
        b'\x4d\x5a',  # DOS/Windows executable header
        b'\x7f\x45\x4c\x46'  # ELF executable header
    ]

    content_lower = file_content.lower()
    for pattern in malicious_patterns:
        if pattern in content_lower:
            return True

    return False

def extract_image_metadata(img_bytes: bytes) -> Dict[str, Any]:
    """Extract comprehensive metadata from image"""
    try:
        image = Image.open(io.BytesIO(img_bytes))

        metadata = {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode,
            'file_size': len(img_bytes)
        }

        # Extract EXIF data
        exif_data = image.getexif()
        if exif_data:
            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if isinstance(value, (str, int, float)):
                    exif_dict[tag_name] = value
            metadata['exif'] = exif_dict

        return metadata

    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        return {}

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

# --- Health and Info Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Palm Oil Data Collection API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "forms": "/api/forms",
            "images": "/api/image-list",
            "harvester_proofs": "/api/harvester-proofs",
            "tree_locations": "/api/tree-locations",
            "analytics": "/api/analytics/dashboard"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": now_ms()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(503, "Service unavailable")

# --- Data Collection Endpoints ---

@app.post("/api/forms")
@limiter.limit("20/minute")
async def create_recon_form(request: Request, form_data: ReconFormRequest):
    """Create a new reconnaissance form with enhanced validation"""
    logger.info(f"Creating reconnaissance form for tree: {form_data.treeId}")

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Check for recent duplicates (within 30 minutes)
            existing_form = await conn.fetchrow("""
                SELECT id FROM recon_forms
                WHERE tree_id = $1 AND plot_id = $2
                AND created_at > $3
                ORDER BY created_at DESC LIMIT 1
            """, form_data.treeId, form_data.plotId, now_ms() - 30*60*1000)

            if existing_form:
                logger.warning(f"Duplicate form detected for tree {form_data.treeId}")
                raise HTTPException(409, "Duplicate form detected within 30 minutes")

            form_id = await conn.fetchval("""
                INSERT INTO recon_forms(
                    tree_id, plot_id, number_of_fruits, harvest_days,
                    created_at, client_id
                ) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id
            """, form_data.treeId, form_data.plotId, form_data.numberOfFruits,
                form_data.harvestDays, now_ms(), form_data.clientId)

            logger.info(f"Form created successfully: {form_id} for tree {form_data.treeId}")
            return {"formId": form_id, "status": "success", "timestamp": now_ms()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Form creation failed: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/api/forms/{plot_id}")
@limiter.limit("100/minute")
async def get_forms_by_plot(
    plot_id: str,
    request: Request,
    include_images: bool = Query(False),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get all forms for a specific plot with optional images"""
    logger.info(f"Retrieving forms for plot: {plot_id}")

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if include_images:
                query = """
                    SELECT f.*,
                           COALESCE(
                               json_agg(
                                   json_build_object(
                                       'id', i.id,
                                       'url', i.url,
                                       'filename', i.filename,
                                       'uploaded_at', i.uploaded_at,
                                       'image_index', i.image_index
                                   ) ORDER BY i.image_index
                               ) FILTER (WHERE i.id IS NOT NULL),
                               '[]'::json
                           ) as images
                    FROM recon_forms f
                    LEFT JOIN images i ON f.id = i.form_id AND i.form_type = 'recon'
                    WHERE f.plot_id = $1
                    GROUP BY f.id
                    ORDER BY f.created_at DESC
                    LIMIT $2 OFFSET $3
                """
                rows = await conn.fetch(query, plot_id, limit, offset)
            else:
                query = """
                    SELECT * FROM recon_forms
                    WHERE plot_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                """
                rows = await conn.fetch(query, plot_id, limit, offset)

            # Get total count
            total = await conn.fetchval("SELECT COUNT(*) FROM recon_forms WHERE plot_id = $1", plot_id)

            forms = [dict(row) for row in rows]
            return {
                "forms": forms,
                "total": total,
                "hasMore": offset + len(forms) < total
            }

    except Exception as e:
        logger.error(f"Failed to retrieve forms for plot {plot_id}: {e}")
        raise HTTPException(500, "Internal server error")

@app.post("/api/harvester-proofs")
@limiter.limit("50/minute")
async def create_harvester_proof(request: Request, proof_data: HarvesterProofRequest):
    """Create a harvester proof record with image and location data"""
    logger.info(f"Creating harvester proof for tree: {proof_data.treeId}")

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            proof_id = await conn.fetchval("""
                INSERT INTO harvester_proofs(
                    tree_id, plot_id, image_url, created_at, updated_at,
                    location_latitude, location_longitude, location_accuracy,
                    notes, harvester_id, client_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING id
            """, proof_data.treeId, proof_data.plotId, proof_data.imageUrl,
                now_ms(), now_ms(), proof_data.latitude, proof_data.longitude,
                proof_data.accuracy, proof_data.notes, proof_data.harvesterId,
                proof_data.clientId)

            logger.info(f"Harvester proof created: {proof_id} for tree {proof_data.treeId}")
            return {"proofId": proof_id, "status": "success", "timestamp": now_ms()}

    except Exception as e:
        logger.error(f"Harvester proof creation failed: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/api/tree-locations")
@limiter.limit("100/minute")
async def create_tree_location(request: Request, location_data: TreeLocationRequest):
    """Create or update tree location with virtual map and GPS coordinates"""
    logger.info(f"Creating/updating location for tree: {location_data.treeId}")

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Check for existing location
            existing_location = await conn.fetchrow("""
                SELECT id, version FROM tree_locations
                WHERE tree_id = $1 AND plot_id = $2
            """, location_data.treeId, location_data.plotId)

            if existing_location:
                # Update existing location
                await conn.execute("""
                    UPDATE tree_locations SET
                        x_coordinate = $3, y_coordinate = $4,
                        latitude = $5, longitude = $6, gps_accuracy = $7,
                        notes = $8, surveyor_id = $9, updated_at = $10,
                        version = version + 1, client_id = $11
                    WHERE id = $1
                """, existing_location['id'], location_data.treeId,
                    location_data.xCoordinate, location_data.yCoordinate,
                    location_data.latitude, location_data.longitude,
                    location_data.accuracy, location_data.notes,
                    location_data.surveyorId, now_ms(), location_data.clientId)

                operation = "updated"
                location_id = existing_location['id']
            else:
                # Create new location
                location_id = await conn.fetchval("""
                    INSERT INTO tree_locations(
                        tree_id, plot_id, x_coordinate, y_coordinate,
                        latitude, longitude, gps_accuracy, notes,
                        surveyor_id, created_at, updated_at, client_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) RETURNING id
                """, location_data.treeId, location_data.plotId,
                    location_data.xCoordinate, location_data.yCoordinate,
                    location_data.latitude, location_data.longitude,
                    location_data.accuracy, location_data.notes,
                    location_data.surveyorId, now_ms(), now_ms(),
                    location_data.clientId)

                operation = "created"

            logger.info(f"Tree location {operation}: {location_id} for tree {location_data.treeId}")
            return {
                "locationId": location_id,
                "status": "success",
                "operation": operation,
                "timestamp": now_ms()
            }

    except Exception as e:
        logger.error(f"Tree location operation failed: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/api/analytics/dashboard")
@limiter.limit("50/minute")
async def get_analytics_dashboard(
    request: Request,
    plot_id: Optional[str] = Query(None),
    date_from: Optional[int] = Query(None),
    date_to: Optional[int] = Query(None)
):
    """Provides analytics data for dashboard visualization"""
    logger.info("Generating analytics dashboard")

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Base conditions
            conditions = []
            params = []

            if plot_id:
                conditions.append(f"plot_id = ${len(params) + 1}")
                params.append(plot_id)
            if date_from:
                conditions.append(f"created_at >= ${len(params) + 1}")
                params.append(date_from)
            if date_to:
                conditions.append(f"created_at <= ${len(params) + 1}")
                params.append(date_to)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            # Summary statistics
            summary_query = f"""
                WITH form_stats AS (
                    SELECT
                        COUNT(*) as total_forms,
                        COUNT(DISTINCT tree_id) as unique_trees,
                        COUNT(DISTINCT plot_id) as unique_plots,
                        AVG(number_of_fruits) as avg_fruits,
                        COUNT(*) FILTER (WHERE is_synced = false) as unsynced_count
                    FROM recon_forms {where_clause}
                ),
                image_stats AS (
                    SELECT COUNT(*) as total_images
                    FROM images i
                    JOIN recon_forms f ON i.form_id = f.id
                    {where_clause.replace('WHERE', 'WHERE' if not where_clause else 'AND')}
                )
                SELECT
                    json_build_object(
                        'summary', (SELECT row_to_json(form_stats) FROM form_stats),
                        'images', (SELECT row_to_json(image_stats) FROM image_stats)
                    ) as dashboard_data
            """

            result = await conn.fetchrow(summary_query, *params)
            dashboard_data = result['dashboard_data']

            return dashboard_data

    except Exception as e:
        logger.error(f"Dashboard analytics failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.post("/api/image-list")
@limiter.limit("10/minute")
async def process_blob_images(request: Request, image_data: ImageListRequest):
    """Process a list of images already uploaded to blob storage with enhanced validation"""
    logger.info(f"Processing {len(image_data.images)} blob images")

    try:
        processed = 0
        errors: List[Dict[str, Any]] = []
        associations = []

        for i, item in enumerate(image_data.images):
            try:
                logger.debug(f"Processing image {i+1}/{len(image_data.images)}")

                if not item.url:
                    errors.append({"index": i, "reason": "no URL provided"})
                    continue

                # Download and validate image
                img_bytes = await download_image_from_url(item.url)
                if not img_bytes:
                    errors.append({"index": i, "reason": "failed to download", "url": item.url})
                    continue

                # Validate image file
                try:
                    await validate_uploaded_file(img_bytes, item.filename or f"image_{i}.jpg")
                except ValueError as e:
                    errors.append({"index": i, "reason": f"validation failed: {str(e)}", "url": item.url})
                    continue

                # Extract comprehensive metadata
                metadata = extract_image_metadata(img_bytes)
                checksum = item.checksum or sha256(img_bytes)

                # Extract timestamp from EXIF or use fallback
                image_timestamp = extract_image_timestamp(img_bytes) or item.timestamp or now_ms()

                # Find form based on timestamp
                form_id = await find_form_by_timestamp(image_timestamp)

                if not form_id:
                    # Create UNMATCHED placeholder
                    unmatched_form = await create_placeholder("UNMATCHED")
                    form_id = unmatched_form
                    tree_id = "UNMATCHED"
                    errors.append({
                        "index": i,
                        "reason": "no matching form found; stored as UNMATCHED",
                        "url": item.url,
                        "timestamp": image_timestamp
                    })
                else:
                    # Get tree ID for association
                    pool = await get_pool()
                    async with pool.acquire() as conn:
                        form_data = await conn.fetchrow("SELECT tree_id FROM recon_forms WHERE id=$1", form_id)
                        tree_id = form_data['tree_id'] if form_data else 'UNKNOWN'

                # Check for duplicates
                pool = await get_pool()
                async with pool.acquire() as conn:
                    existing = await conn.fetchrow(
                        "SELECT id FROM images WHERE form_id=$1 AND checksum=$2",
                        form_id, checksum
                    )

                    if existing:
                        errors.append({"index": i, "reason": "duplicate image", "url": item.url})
                        continue

                    # Insert enhanced image record
                    image_id = await conn.fetchval("""
                        INSERT INTO images(
                            form_id, form_type, url, filename, original_filename,
                            checksum, file_size, mime_type, image_width, image_height,
                            exif_data, uploaded_at, processing_status
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13) RETURNING id
                    """, form_id, "recon", item.url, item.filename, item.filename,
                        checksum, metadata.get('file_size'), magic.from_buffer(img_bytes, mime=True) if magic else None,
                        metadata.get('width'), metadata.get('height'),
                        metadata.get('exif'), now_ms(), "processed")

                associations.append({
                    "imageId": image_id,
                    "formId": form_id,
                    "treeId": tree_id
                })
                processed += 1

            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                errors.append({"index": i, "error": str(e)})

        logger.info(f"Image processing complete: {processed} processed, {len(errors)} errors")
        return {
            "processed": processed,
            "errors": errors,
            "associations": associations
        }

    except Exception as e:
        logger.error(f"Image list processing failed: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")
