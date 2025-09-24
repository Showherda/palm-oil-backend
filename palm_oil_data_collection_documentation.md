# Palm Oil Data Collection System: Comprehensive Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Collection Architecture](#data-collection-architecture)
3. [Recon Module Data Flow](#recon-module-data-flow)
4. [Harvester Module Data Flow](#harvester-module-data-flow)
5. [Data Retention & Storage](#data-retention--storage)
6. [Backend Requirements](#backend-requirements)
7. [API Specifications](#api-specifications)
8. [Data Synchronization](#data-synchronization)
9. [Security & Validation](#security--validation)
10. [Performance Considerations](#performance-considerations)

---

## System Overview

The Palm Oil Data Collection System is an offline-first mobile application that collects reconnaissance and harvester data for palm oil tree management. The system operates in two primary modes:

### **Recon Mode**: Tree Survey & Assessment
- Collects detailed tree information, fruit counts, and harvest timing
- Captures photographic evidence (up to 3 images per form)
- Records tree and plot identifiers for tracking

### **Harvester Mode**: Proof of Work & Location Mapping
- Documents harvester activities with photo evidence
- Records precise GPS coordinates for tree locations
- Manages virtual map coordinates for plot visualization
- Tracks harvester notes and timestamps

### **Architecture Principles**
- **Offline-First**: All data collected locally, synced when network available
- **Data Integrity**: Comprehensive validation and duplicate prevention
- **Scalability**: Designed for thousands of trees across multiple plots
- **Audit Trail**: Complete timestamp and sync tracking

---

## Data Collection Architecture

### Frontend Mobile App (Android - Kotlin)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Entry    │    │  Local Storage  │    │  Sync Manager   │
│    Activities   │───▶│   (SQLite)      │───▶│   (Network)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Camera/Media   │    │  Room Database  │    │ Backend API     │
│   Management    │    │   Operations    │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Backend Infrastructure (FastAPI + PostgreSQL)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Business Logic │    │   Database      │
│  (FastAPI)      │───▶│   Validation    │───▶│ (PostgreSQL)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Blob Storage   │    │   Rate Limiting │    │   Data Analytics│
│  (Vercel)       │    │  & Security     │    │   & Reporting   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Recon Module Data Flow

### Frontend Data Collection Process

#### 1. **Form Initiation** (`ReconForm.kt`)
```kotlin
// User Input
- tree_id: String (required) // Primary tree identifier
```
**Frontend Actions:**
- Validates tree ID format and length
- Navigates to detailed capture form
- Passes tree ID to next activity via Intent

**Backend Expectation:**
- No immediate action required
- Prepare for incoming form data

#### 2. **Detailed Data Capture** (`ReconFormCapture.kt`)
```kotlin
// Data Collection Fields
data class ReconFormData {
    val treeId: String,           // From previous screen
    val plotId: String,           // User input (required)
    val numberOfFruits: Int,      // User input (required, > 0)
    val harvestDays: Int,         // Radio selection (1, 2, or 3)
    val images: List<String>      // Up to 3 image file paths
}
```

**Frontend Actions:**
- **Validation**: Ensures all required fields are filled
- **Image Management**: Captures up to 3 photos via camera
- **Local Storage**: Saves to SQLite with `is_synced = false`
- **Error Handling**: Provides user feedback for validation failures

**Backend Expectation:**
- Receive validated form data via `/api/forms` endpoint
- Handle image URLs from blob storage via `/api/image-list` endpoint

#### 3. **Image Capture Process** (`ReconCamera.kt`)
```kotlin
// Image Processing Pipeline
1. Camera capture with CameraX
2. File storage: "/external/files/IMG_timestamp.jpg"
3. Return file path to form
4. Display preview in form
```

**Frontend Actions:**
- **Quality Control**: Ensures images are properly captured and stored
- **File Management**: Creates timestamped filenames to prevent conflicts
- **Memory Management**: Optimizes image loading for performance

**Backend Expectation:**
- Process images from Vercel Blob Storage URLs
- Extract EXIF metadata for timestamp correlation
- Associate images with correct forms based on timestamps

#### 4. **Local Data Storage** (`ReconFormEntity.kt`)
```sql
-- Frontend SQLite Schema
CREATE TABLE recon_forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tree_id TEXT NOT NULL,
    plot_id TEXT NOT NULL,
    number_of_fruits INTEGER NOT NULL,
    harvest_days INTEGER NOT NULL,
    image1_path TEXT,
    image2_path TEXT,
    image3_path TEXT,
    created_at INTEGER NOT NULL,
    is_synced INTEGER DEFAULT 0
);
```

**Frontend Actions:**
- **Atomic Operations**: Uses Room transactions for data integrity
- **Indexing**: Optimizes queries with indexed fields
- **Sync Tracking**: Maintains sync status for each record

**Backend Expectation:**
- Mirror database structure with enhanced PostgreSQL features
- Implement proper foreign key relationships
- Add audit fields for compliance

#### 5. **Data Synchronization** (`ReconUpload.kt`)

**Frontend Sync Process:**
```kotlin
// Upload Workflow
1. Query unsynced forms: `SELECT * FROM recon_forms WHERE is_synced = 0`
2. Upload images to Vercel Blob Storage
3. Send form data to `/api/forms` endpoint
4. Send image URLs to `/api/image-list` endpoint
5. Mark local records as synced: `UPDATE recon_forms SET is_synced = 1`
```

**Frontend Actions:**
- **Batch Processing**: Handles multiple forms efficiently
- **Error Recovery**: Retries failed uploads with exponential backoff
- **Progress Tracking**: Provides user feedback during uploads
- **Network Awareness**: Only syncs when connectivity available

**Backend Expected Actions:**
- **Form Processing**: Validate and store form data in PostgreSQL
- **Image Association**: Download images from blob URLs and associate with forms
- **Duplicate Prevention**: Check for existing records using checksums
- **Response Handling**: Return success/failure status with detailed errors

### Frontend Recon Viewing & Management

#### 6. **Form Viewing** (`ReconViewForm.kt`)
**Frontend Actions:**
- **Data Retrieval**: Loads all forms with LiveData for reactive updates
- **Image Display**: Shows thumbnails of associated images
- **Filtering**: Allows search by tree ID, plot ID, date ranges
- **Status Indication**: Visual indicators for sync status

**Backend Expected Actions:**
- **Data Serving**: Provide API endpoints for form retrieval
- **Pagination**: Handle large datasets efficiently
- **Search**: Support complex queries with multiple filters

#### 7. **Gallery Management** (`ReconGallery.kt`)
**Frontend Actions:**
- **Image Aggregation**: Collects all images across all forms
- **Metadata Display**: Shows tree ID, plot ID, timestamps for each image
- **Full-Screen Viewing**: Provides detailed image examination
- **Batch Operations**: Enables bulk image management

**Backend Expected Actions:**
- **Image Serving**: Provide optimized image URLs for gallery display
- **Metadata API**: Return comprehensive image metadata
- **Performance**: Implement CDN and caching for image delivery

---

## Harvester Module Data Flow

### Frontend Data Collection Process

#### 1. **Harvester Proof Collection** (`HarvesterProofEntity.kt`)
```kotlin
data class HarvesterProofEntity(
    val id: Long = 0,
    val treeId: String,              // Tree identifier
    val plotId: String,              // Plot identifier
    val imagePath: String,           // Single proof image
    val createdAt: Long,             // Timestamp
    val updatedAt: Long,             // Last modification
    val isSynced: Boolean = false,   // Sync status
    val syncTimestamp: Long? = null, // When synced
    val locationLatitude: Double?,   // GPS coordinates
    val locationLongitude: Double?,
    val notes: String? = null        // Additional notes
)
```

**Frontend Actions:**
- **GPS Integration**: Automatically captures location when available
- **Image Validation**: Ensures single high-quality proof image
- **Metadata Enrichment**: Adds timestamps and location data
- **Note Management**: Allows harvester to add contextual information

**Backend Expected Actions:**
- **Location Validation**: Verify GPS coordinates are within expected bounds
- **Image Processing**: Store and optimize proof images
- **Audit Trail**: Maintain complete history of harvester activities
- **Geospatial Queries**: Enable location-based search and analysis

#### 2. **Tree Location Management** (`TreeLocationEntity.kt`)
```kotlin
data class TreeLocationEntity(
    val id: Long = 0,
    val treeId: String,              // Tree identifier
    val plotId: String,              // Plot identifier
    val xCoordinate: Float,          // Virtual map X position
    val yCoordinate: Float,          // Virtual map Y position
    val latitude: Double? = null,    // Real GPS latitude
    val longitude: Double? = null,   // Real GPS longitude
    val createdAt: Long,             // Creation timestamp
    val updatedAt: Long,             // Last update timestamp
    val isSynced: Boolean = false,   // Sync status
    val syncTimestamp: Long? = null, // Sync completion time
    val notes: String? = null        // Location notes
)
```

**Frontend Actions:**
- **Coordinate Management**: Handles both virtual map and real GPS coordinates
- **Position Validation**: Ensures coordinates are within plot boundaries
- **Update Tracking**: Maintains history of location changes
- **Constraint Enforcement**: Prevents duplicate trees in same plot

**Backend Expected Actions:**
- **Coordinate Validation**: Verify both virtual and GPS coordinates
- **Spatial Indexing**: Implement PostGIS for geospatial operations
- **Boundary Checking**: Validate coordinates against plot boundaries
- **Conflict Resolution**: Handle coordinate update conflicts

#### 3. **Virtual Map Integration** (`HarvesterVirtualMapView.kt`, `HarvesterDownloadMap.kt`)
**Frontend Actions:**
- **Map Rendering**: Displays trees on virtual plot maps
- **Interactive Positioning**: Allows drag-and-drop tree positioning
- **Real-time Updates**: Synchronizes changes across map views
- **Export Functionality**: Generates downloadable map data

**Backend Expected Actions:**
- **Map Data API**: Provide plot boundaries and tree positions
- **Real-time Sync**: Support WebSocket or polling for live updates
- **Export Generation**: Create downloadable map files (KML, GeoJSON)
- **Version Control**: Track map changes over time

---

## Data Retention & Storage

### Frontend Storage Strategy

#### **Local SQLite Database** (`PalmOilDatabase.kt`)
```kotlin
@Database(
    entities = [
        ReconFormEntity::class,
        HarvesterProofEntity::class,
        TreeLocationEntity::class
    ],
    version = 4,
    exportSchema = true
)
```

**Retention Policies:**
- **Unsynced Data**: Retained indefinitely until successful upload
- **Synced Data**: Configurable retention (default: 30 days)
- **Images**: Local copies deleted after successful cloud upload
- **Cache**: Automatic cleanup based on storage constraints

**Storage Optimization:**
- **Compression**: Image compression before storage
- **Indexing**: Strategic database indexes for performance
- **Cleanup**: Automatic removal of orphaned records
- **Backup**: Local backup before major operations

### Backend Storage Strategy

#### **PostgreSQL Primary Storage**
```sql
-- Data Retention Configuration
CREATE TABLE data_retention_policies (
    table_name VARCHAR(50) PRIMARY KEY,
    retention_days INTEGER NOT NULL,
    archive_enabled BOOLEAN DEFAULT true,
    last_cleanup TIMESTAMP DEFAULT NOW()
);

INSERT INTO data_retention_policies VALUES
('recon_forms', 365, true, NOW()),           -- 1 year retention
('harvester_proofs', 730, true, NOW()),      -- 2 years retention
('tree_locations', -1, false, NOW()),        -- Permanent retention
('images', 365, true, NOW());                -- 1 year retention
```

**Backup Strategy:**
- **Daily Backups**: Automated full database backups
- **Point-in-Time Recovery**: WAL archiving for precise recovery
- **Geographic Redundancy**: Multi-region backup storage
- **Testing**: Regular backup restoration testing

#### **Blob Storage Management** (Vercel/AWS S3)
**Storage Classes:**
- **Hot Storage**: Recent images (0-30 days)
- **Warm Storage**: Older images (30-365 days)
- **Cold Storage**: Archive images (1+ years)
- **Glacier**: Long-term compliance storage (5+ years)

**Lifecycle Policies:**
```json
{
  "lifecycle_rules": [
    {
      "name": "transition_to_warm",
      "days": 30,
      "storage_class": "STANDARD_IA"
    },
    {
      "name": "transition_to_cold",
      "days": 365,
      "storage_class": "GLACIER"
    },
    {
      "name": "delete_expired",
      "days": 2555,  // 7 years
      "action": "DELETE"
    }
  ]
}
```

---

## Backend Requirements

### Core Infrastructure Requirements

#### **1. Database Schema** (PostgreSQL 13+)
```sql
-- Enhanced recon_forms table
CREATE TABLE recon_forms (
    id SERIAL PRIMARY KEY,
    tree_id VARCHAR(50) NOT NULL,
    plot_id VARCHAR(50) NOT NULL,
    number_of_fruits INTEGER NOT NULL CHECK (number_of_fruits > 0),
    harvest_days INTEGER NOT NULL CHECK (harvest_days IN (1, 2, 3)),
    created_at BIGINT NOT NULL,
    updated_at BIGINT DEFAULT EXTRACT(EPOCH FROM NOW()) * 1000,
    is_synced BOOLEAN DEFAULT false,
    sync_timestamp BIGINT,
    client_id UUID, -- For conflict resolution
    version INTEGER DEFAULT 1, -- For optimistic locking

    -- Audit fields
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    -- Indexes
    INDEX idx_recon_forms_tree_id (tree_id),
    INDEX idx_recon_forms_plot_id (plot_id),
    INDEX idx_recon_forms_created_at (created_at),
    INDEX idx_recon_forms_sync_status (is_synced, sync_timestamp),
    INDEX idx_recon_forms_composite (tree_id, plot_id, created_at)
);

-- Images table with enhanced metadata
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    form_id INTEGER REFERENCES recon_forms(id) ON DELETE CASCADE,
    form_type VARCHAR(20) NOT NULL, -- 'recon' or 'harvester'
    url TEXT NOT NULL,
    filename VARCHAR(255),
    original_filename VARCHAR(255),
    checksum VARCHAR(64) NOT NULL, -- SHA-256
    file_size BIGINT,
    mime_type VARCHAR(50),
    image_width INTEGER,
    image_height INTEGER,
    exif_data JSONB, -- EXIF metadata
    uploaded_at BIGINT NOT NULL,
    image_index INTEGER CHECK (image_index BETWEEN 1 AND 3),
    processing_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processed', 'error'

    -- Constraints
    UNIQUE (form_id, image_index, form_type),
    INDEX idx_images_form_id (form_id),
    INDEX idx_images_checksum (checksum),
    INDEX idx_images_upload_date (uploaded_at),
    INDEX idx_images_processing_status (processing_status)
);

-- Harvester proofs table
CREATE TABLE harvester_proofs (
    id SERIAL PRIMARY KEY,
    tree_id VARCHAR(50) NOT NULL,
    plot_id VARCHAR(50) NOT NULL,
    image_url TEXT, -- Cloud storage URL
    local_image_path TEXT, -- Original local path reference
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    is_synced BOOLEAN DEFAULT false,
    sync_timestamp BIGINT,
    location_latitude DECIMAL(10, 8), -- 8 decimal places for ~1mm precision
    location_longitude DECIMAL(11, 8),
    location_accuracy DECIMAL(6, 2), -- GPS accuracy in meters
    notes TEXT,
    harvester_id VARCHAR(100), -- Who performed the work

    -- Audit fields
    client_id UUID,
    version INTEGER DEFAULT 1,

    -- Indexes
    INDEX idx_harvester_proofs_tree_id (tree_id),
    INDEX idx_harvester_proofs_plot_id (plot_id),
    INDEX idx_harvester_proofs_location (location_latitude, location_longitude),
    INDEX idx_harvester_proofs_sync (is_synced, sync_timestamp),
    INDEX idx_harvester_proofs_harvester (harvester_id, created_at)
);

-- Tree locations table with spatial support
CREATE TABLE tree_locations (
    id SERIAL PRIMARY KEY,
    tree_id VARCHAR(50) NOT NULL,
    plot_id VARCHAR(50) NOT NULL,

    -- Virtual map coordinates (for plot visualization)
    x_coordinate DECIMAL(10, 6) NOT NULL,
    y_coordinate DECIMAL(10, 6) NOT NULL,

    -- Real GPS coordinates
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    gps_accuracy DECIMAL(6, 2), -- GPS accuracy in meters

    -- Timestamps
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    is_synced BOOLEAN DEFAULT false,
    sync_timestamp BIGINT,

    -- Additional metadata
    notes TEXT,
    surveyor_id VARCHAR(100), -- Who recorded the location

    -- Audit fields
    client_id UUID,
    version INTEGER DEFAULT 1,

    -- Spatial column (requires PostGIS)
    geom GEOMETRY(POINT, 4326), -- WGS84 coordinate system

    -- Constraints
    UNIQUE (plot_id, tree_id), -- One location per tree per plot

    -- Indexes
    INDEX idx_tree_locations_tree_id (tree_id),
    INDEX idx_tree_locations_plot_id (plot_id),
    INDEX idx_tree_locations_coordinates (x_coordinate, y_coordinate),
    INDEX idx_tree_locations_gps (latitude, longitude),
    INDEX idx_tree_locations_sync (is_synced, sync_timestamp),
    SPATIAL INDEX idx_tree_locations_geom (geom)
);

-- Data quality and analytics tables
CREATE TABLE sync_logs (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    operation VARCHAR(20) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    client_id UUID NOT NULL,
    sync_timestamp BIGINT NOT NULL,
    payload_size INTEGER,
    processing_time_ms INTEGER,
    status VARCHAR(20) NOT NULL, -- 'success', 'error', 'partial'
    error_message TEXT,

    INDEX idx_sync_logs_timestamp (sync_timestamp),
    INDEX idx_sync_logs_table (table_name, record_id),
    INDEX idx_sync_logs_client (client_id),
    INDEX idx_sync_logs_status (status)
);
```

#### **2. API Performance Requirements**

**Response Time SLAs:**
- Form submission: < 500ms (95th percentile)
- Image upload: < 2s per image (95th percentile)
- Data retrieval: < 200ms (95th percentile)
- Bulk operations: < 5s for 100 records (95th percentile)

**Throughput Requirements:**
- Forms: 1000 submissions/hour/device
- Images: 500 uploads/hour/device
- Queries: 10,000 requests/hour/device
- Concurrent users: 100+ simultaneous devices

**Reliability Requirements:**
- Uptime: 99.9% availability
- Data consistency: ACID compliance
- Backup recovery: RTO < 1 hour, RPO < 15 minutes
- Error rate: < 0.1% for all operations

#### **3. Security & Compliance Requirements**

**Authentication & Authorization:**
```python
# API Key-based authentication (current)
# Future: JWT with role-based access control

class SecurityConfig:
    API_KEY_ROTATION_DAYS = 90
    JWT_EXPIRATION_HOURS = 24
    RATE_LIMIT_PER_MINUTE = 100
    MAX_UPLOAD_SIZE_MB = 50
    ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
```

**Data Protection:**
- **Encryption at Rest**: AES-256 for database and storage
- **Encryption in Transit**: TLS 1.3 for all API communications
- **PII Handling**: Minimal collection, secure hashing where possible
- **Audit Logging**: Complete audit trail for all data operations

**Input Validation:**
```python
# Comprehensive validation rules
from pydantic import BaseModel, validator
from typing import Optional, List

class ReconFormRequest(BaseModel):
    treeId: str
    plotId: str
    numberOfFruits: int
    harvestDays: int

    @validator('treeId')
    def validate_tree_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Tree ID is required')
        if len(v) > 50:
            raise ValueError('Tree ID too long')
        if not re.match(r'^[A-Za-z0-9_-]+$', v):
            raise ValueError('Tree ID contains invalid characters')
        return v.strip()

    @validator('numberOfFruits')
    def validate_fruit_count(cls, v):
        if v <= 0:
            raise ValueError('Number of fruits must be positive')
        if v > 10000:
            raise ValueError('Number of fruits seems unrealistic')
        return v

    @validator('harvestDays')
    def validate_harvest_days(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError('Harvest days must be 1, 2, or 3')
        return v
```

---

## API Specifications

### **Core Recon APIs**

#### **POST /api/forms** - Create Recon Form
```python
@app.post("/api/forms")
@limiter.limit("20/minute")
async def create_recon_form(request: ReconFormRequest):
    """
    Creates a new reconnaissance form entry.

    Request Body:
    {
        "treeId": "string (required, max 50 chars)",
        "plotId": "string (required, max 50 chars)",
        "numberOfFruits": "integer (required, > 0)",
        "harvestDays": "integer (required, 1-3)"
    }

    Response:
    {
        "formId": "integer",
        "status": "success",
        "timestamp": "bigint"
    }

    Error Response:
    {
        "error": "string",
        "code": "string",
        "details": "object"
    }
    """
```

**Backend Implementation Requirements:**
```python
async def create_recon_form(request: ReconFormRequest):
    # 1. Input validation (Pydantic handles this)
    # 2. Duplicate detection
    existing_form = await check_duplicate_form(
        request.treeId,
        request.plotId,
        time_window_minutes=30
    )
    if existing_form:
        raise HTTPException(409, "Duplicate form detected")

    # 3. Database insertion with transaction
    async with get_db_transaction() as conn:
        form_id = await conn.fetchval("""
            INSERT INTO recon_forms
            (tree_id, plot_id, number_of_fruits, harvest_days, created_at, client_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """, request.treeId, request.plotId, request.numberOfFruits,
            request.harvestDays, now_ms(), request.client_id)

        # 4. Log the operation
        await log_sync_operation("recon_forms", form_id, "INSERT",
                                request.client_id, "success")

    # 5. Return response
    return {"formId": form_id, "status": "success", "timestamp": now_ms()}
```

#### **POST /api/image-list** - Process Uploaded Images
```python
@app.post("/api/image-list")
@limiter.limit("10/minute")
async def process_image_list(request: ImageListRequest):
    """
    Processes a batch of images already uploaded to blob storage.
    Associates images with forms based on timestamp matching.

    Request Body:
    {
        "images": [
            {
                "url": "string (required, HTTPS blob URL)",
                "filename": "string (optional)",
                "timestamp": "bigint (optional, fallback timestamp)",
                "checksum": "string (optional, for duplicate detection)"
            }
        ]
    }

    Response:
    {
        "processed": "integer",
        "errors": [
            {
                "index": "integer",
                "reason": "string",
                "url": "string"
            }
        ],
        "associations": [
            {
                "imageId": "integer",
                "formId": "integer",
                "treeId": "string"
            }
        ]
    }
    """
```

**Backend Implementation Requirements:**
```python
async def process_image_list(request: ImageListRequest):
    processed = 0
    errors = []
    associations = []

    for idx, image_item in enumerate(request.images):
        try:
            # 1. Validate blob URL
            if not validate_blob_url(image_item.url):
                errors.append({
                    "index": idx,
                    "reason": "Invalid or unauthorized blob URL",
                    "url": image_item.url
                })
                continue

            # 2. Download and analyze image
            image_bytes = await download_image_from_url(image_item.url)
            if not image_bytes:
                errors.append({
                    "index": idx,
                    "reason": "Failed to download image",
                    "url": image_item.url
                })
                continue

            # 3. Extract metadata
            checksum = sha256(image_bytes)
            exif_data = extract_exif_metadata(image_bytes)
            image_timestamp = extract_image_timestamp(image_bytes) or image_item.timestamp

            # 4. Find associated form
            form_id = await find_form_by_timestamp(image_timestamp)
            if not form_id:
                # Create placeholder or mark as unmatched
                form_id = await create_placeholder_form("UNMATCHED")
                errors.append({
                    "index": idx,
                    "reason": "No matching form found, stored as UNMATCHED",
                    "url": image_item.url,
                    "timestamp": image_timestamp
                })

            # 5. Check for duplicates
            existing_image = await check_duplicate_image(checksum, form_id)
            if existing_image:
                errors.append({
                    "index": idx,
                    "reason": "Duplicate image detected",
                    "url": image_item.url
                })
                continue

            # 6. Store image metadata
            async with get_db_transaction() as conn:
                image_id = await conn.fetchval("""
                    INSERT INTO images
                    (form_id, form_type, url, filename, checksum,
                     file_size, exif_data, uploaded_at, processing_status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, form_id, "recon", image_item.url, image_item.filename,
                    checksum, len(image_bytes), exif_data, now_ms(), "processed")

                # Get form details for response
                form_data = await conn.fetchrow(
                    "SELECT tree_id FROM recon_forms WHERE id = $1", form_id
                )

            associations.append({
                "imageId": image_id,
                "formId": form_id,
                "treeId": form_data['tree_id'] if form_data else "UNMATCHED"
            })
            processed += 1

        except Exception as e:
            errors.append({
                "index": idx,
                "reason": f"Processing error: {str(e)}",
                "url": image_item.url
            })

    return {
        "processed": processed,
        "errors": errors,
        "associations": associations
    }
```

### **Harvester APIs**

#### **POST /api/harvester-proofs** - Create Harvester Proof
```python
@app.post("/api/harvester-proofs")
@limiter.limit("50/minute")
async def create_harvester_proof(request: HarvesterProofRequest):
    """
    Creates a harvester proof record with image and location data.

    Request Body:
    {
        "treeId": "string (required)",
        "plotId": "string (required)",
        "imageUrl": "string (required, blob storage URL)",
        "latitude": "number (optional, GPS)",
        "longitude": "number (optional, GPS)",
        "accuracy": "number (optional, GPS accuracy in meters)",
        "notes": "string (optional)",
        "harvesterId": "string (optional)"
    }

    Response:
    {
        "proofId": "integer",
        "status": "success",
        "timestamp": "bigint"
    }
    """
```

#### **POST /api/tree-locations** - Create/Update Tree Location
```python
@app.post("/api/tree-locations")
@limiter.limit("100/minute")
async def create_tree_location(request: TreeLocationRequest):
    """
    Creates or updates a tree location with virtual map and GPS coordinates.

    Request Body:
    {
        "treeId": "string (required)",
        "plotId": "string (required)",
        "xCoordinate": "number (required, virtual map X)",
        "yCoordinate": "number (required, virtual map Y)",
        "latitude": "number (optional, GPS)",
        "longitude": "number (optional, GPS)",
        "accuracy": "number (optional, GPS accuracy)",
        "notes": "string (optional)",
        "surveyorId": "string (optional)"
    }

    Response:
    {
        "locationId": "integer",
        "status": "success",
        "operation": "created|updated",
        "timestamp": "bigint"
    }
    """
```

**Backend Implementation Notes:**
```python
async def create_tree_location(request: TreeLocationRequest):
    # 1. Check for existing location
    existing_location = await get_tree_location(request.treeId, request.plotId)

    if existing_location:
        # Update existing location
        operation = "updated"
        location_id = await update_tree_location(existing_location.id, request)
    else:
        # Create new location
        operation = "created"
        location_id = await insert_tree_location(request)

    # 2. Update spatial index if GPS coordinates provided
    if request.latitude and request.longitude:
        await update_spatial_index(location_id, request.latitude, request.longitude)

    # 3. Validate coordinates are within plot boundaries
    if not await validate_coordinates_in_plot(request.plotId,
                                            request.xCoordinate,
                                            request.yCoordinate):
        raise HTTPException(400, "Coordinates outside plot boundaries")

    return {
        "locationId": location_id,
        "status": "success",
        "operation": operation,
        "timestamp": now_ms()
    }
```

### **Query APIs**

#### **GET /api/forms/{treeId}** - Get Forms by Tree
```python
@app.get("/api/forms/{tree_id}")
async def get_forms_by_tree(tree_id: str,
                           include_images: bool = False,
                           limit: int = 100,
                           offset: int = 0):
    """
    Retrieves all recon forms for a specific tree.

    Query Parameters:
    - include_images: boolean (include image URLs)
    - limit: integer (max 1000)
    - offset: integer (for pagination)

    Response:
    {
        "forms": [
            {
                "id": "integer",
                "treeId": "string",
                "plotId": "string",
                "numberOfFruits": "integer",
                "harvestDays": "integer",
                "createdAt": "bigint",
                "syncedAt": "bigint",
                "images": [
                    {
                        "id": "integer",
                        "url": "string",
                        "filename": "string",
                        "uploadedAt": "bigint"
                    }
                ]
            }
        ],
        "total": "integer",
        "hasMore": "boolean"
    }
    """
```

#### **GET /api/analytics/dashboard** - Analytics Dashboard
```python
@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard(plot_id: Optional[str] = None,
                                date_from: Optional[int] = None,
                                date_to: Optional[int] = None):
    """
    Provides analytics data for dashboard visualization.

    Response:
    {
        "summary": {
            "totalForms": "integer",
            "totalTrees": "integer",
            "totalPlots": "integer",
            "totalImages": "integer",
            "avgFruitsPerTree": "number",
            "syncStatus": {
                "synced": "integer",
                "pending": "integer"
            }
        },
        "trends": {
            "formsPerDay": [
                {"date": "string", "count": "integer"}
            ],
            "fruitsPerDay": [
                {"date": "string", "total": "integer"}
            ]
        },
        "plots": [
            {
                "plotId": "string",
                "treeCount": "integer",
                "formCount": "integer",
                "lastActivity": "bigint"
            }
        ]
    }
    """
```

---

## Data Synchronization

### Frontend Sync Strategy

#### **Conflict Resolution**
```kotlin
// Client-side conflict detection
data class SyncConflict(
    val localRecord: ReconFormEntity,
    val serverRecord: ReconFormEntity,
    val conflictType: ConflictType,
    val resolutionStrategy: ResolutionStrategy
)

enum class ConflictType {
    VERSION_MISMATCH,    // Different versions of same record
    TIMESTAMP_CONFLICT,  // Same timestamp, different data
    DUPLICATE_DETECTION  // Potential duplicate from different source
}

enum class ResolutionStrategy {
    SERVER_WINS,         // Use server version
    CLIENT_WINS,         // Use client version
    MERGE,              // Attempt automatic merge
    MANUAL_RESOLUTION   // Require user input
}
```

**Frontend Sync Process:**
```kotlin
class SyncManager {
    suspend fun performSync(): SyncResult {
        val syncResult = SyncResult()

        try {
            // 1. Upload unsynced data
            val unsyncedForms = repository.getUnsyncedForms()
            for (form in unsyncedForms) {
                val result = uploadForm(form)
                if (result.success) {
                    repository.markAsSynced(form.id, result.serverTimestamp)
                    syncResult.uploaded.add(form.id)
                } else {
                    syncResult.errors.add(SyncError(form.id, result.error))
                }
            }

            // 2. Download server updates
            val lastSyncTime = getLastSyncTimestamp()
            val serverUpdates = apiService.getUpdates(lastSyncTime)

            for (update in serverUpdates) {
                val conflict = detectConflict(update)
                if (conflict != null) {
                    syncResult.conflicts.add(conflict)
                } else {
                    repository.applyServerUpdate(update)
                    syncResult.downloaded.add(update.id)
                }
            }

            // 3. Update sync timestamp
            setLastSyncTimestamp(now())

        } catch (e: Exception) {
            syncResult.generalError = e.message
        }

        return syncResult
    }
}
```

#### **Backend Sync Management**

**Conflict Detection & Resolution:**
```python
async def handle_sync_conflict(client_record: dict,
                              server_record: dict) -> dict:
    """
    Handles synchronization conflicts between client and server data.

    Resolution Strategy:
    1. Timestamp-based: Most recent wins
    2. Version-based: Higher version wins
    3. Data integrity: Validate both versions
    4. Business rules: Apply domain-specific rules
    """

    conflict_type = detect_conflict_type(client_record, server_record)

    if conflict_type == "VERSION_MISMATCH":
        if client_record['version'] > server_record['version']:
            # Client has newer version
            resolution = await apply_client_version(client_record, server_record)
        else:
            # Server has newer version
            resolution = await apply_server_version(client_record, server_record)

    elif conflict_type == "TIMESTAMP_CONFLICT":
        # Use most recent timestamp
        if client_record['updated_at'] > server_record['updated_at']:
            resolution = await apply_client_version(client_record, server_record)
        else:
            resolution = await apply_server_version(client_record, server_record)

    elif conflict_type == "DATA_INTEGRITY":
        # Validate data integrity and merge if possible
        resolution = await merge_records(client_record, server_record)

    else:
        # Unknown conflict type - default to server wins
        resolution = await apply_server_version(client_record, server_record)

    # Log conflict resolution for audit
    await log_conflict_resolution(conflict_type, resolution)

    return resolution

async def merge_records(client_record: dict, server_record: dict) -> dict:
    """
    Attempts to merge conflicting records using business rules.
    """
    merged_record = server_record.copy()

    # Business rule: Always keep highest fruit count
    if client_record['number_of_fruits'] > server_record['number_of_fruits']:
        merged_record['number_of_fruits'] = client_record['number_of_fruits']
        merged_record['updated_at'] = now_ms()
        merged_record['version'] += 1

    # Business rule: Merge image arrays
    client_images = set(filter(None, [
        client_record.get('image1_path'),
        client_record.get('image2_path'),
        client_record.get('image3_path')
    ]))

    server_images = set(filter(None, [
        server_record.get('image1_path'),
        server_record.get('image2_path'),
        server_record.get('image3_path')
    ]))

    all_images = list(client_images.union(server_images))[:3]

    for i, image_path in enumerate(all_images):
        merged_record[f'image{i+1}_path'] = image_path

    return merged_record
```

### Real-time Sync Features

#### **WebSocket Integration** (Future Enhancement)
```python
# WebSocket endpoint for real-time updates
@app.websocket("/ws/sync/{client_id}")
async def websocket_sync(websocket: WebSocket, client_id: str):
    """
    Real-time synchronization via WebSocket connection.
    Pushes immediate updates to connected clients.
    """
    await websocket.accept()

    # Register client for real-time updates
    sync_manager.register_client(client_id, websocket)

    try:
        while True:
            # Listen for client messages
            message = await websocket.receive_json()

            if message['type'] == 'PING':
                await websocket.send_json({'type': 'PONG'})

            elif message['type'] == 'SUBSCRIBE':
                # Subscribe to specific data updates
                await sync_manager.subscribe_client(
                    client_id,
                    message['topics']
                )

            elif message['type'] == 'DATA_UPDATE':
                # Handle real-time data update
                await handle_realtime_update(message['data'])

    except WebSocketDisconnect:
        sync_manager.unregister_client(client_id)
```

---

## Security & Validation

### Input Validation & Sanitization

#### **Frontend Validation**
```kotlin
class DataValidator {
    fun validateReconForm(form: ReconFormData): ValidationResult {
        val errors = mutableListOf<ValidationError>()

        // Tree ID validation
        if (form.treeId.isBlank()) {
            errors.add(ValidationError("treeId", "Tree ID is required"))
        } else if (form.treeId.length > 50) {
            errors.add(ValidationError("treeId", "Tree ID too long"))
        } else if (!form.treeId.matches(Regex("^[A-Za-z0-9_-]+$"))) {
            errors.add(ValidationError("treeId", "Tree ID contains invalid characters"))
        }

        // Plot ID validation
        if (form.plotId.isBlank()) {
            errors.add(ValidationError("plotId", "Plot ID is required"))
        }

        // Number of fruits validation
        if (form.numberOfFruits <= 0) {
            errors.add(ValidationError("numberOfFruits", "Must be positive"))
        } else if (form.numberOfFruits > 10000) {
            errors.add(ValidationError("numberOfFruits", "Value seems unrealistic"))
        }

        // Harvest days validation
        if (form.harvestDays !in 1..3) {
            errors.add(ValidationError("harvestDays", "Must be 1, 2, or 3"))
        }

        // Image validation
        form.images.forEach { imagePath ->
            if (!File(imagePath).exists()) {
                errors.add(ValidationError("images", "Image file not found: $imagePath"))
            }
        }

        return ValidationResult(errors.isEmpty(), errors)
    }
}
```

#### **Backend Validation & Security**
```python
# Comprehensive input validation with Pydantic
from pydantic import BaseModel, validator, root_validator
import re
from typing import Optional, List

class ReconFormRequest(BaseModel):
    treeId: str
    plotId: str
    numberOfFruits: int
    harvestDays: int
    clientId: Optional[str] = None

    @validator('treeId')
    def validate_tree_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Tree ID is required')
        if len(v) > 50:
            raise ValueError('Tree ID too long (max 50 characters)')
        if not re.match(r'^[A-Za-z0-9_-]+$', v):
            raise ValueError('Tree ID contains invalid characters')
        return v

    @validator('plotId')
    def validate_plot_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Plot ID is required')
        if len(v) > 50:
            raise ValueError('Plot ID too long (max 50 characters)')
        # Check if plot exists in authorized plots
        # if not is_authorized_plot(v):
        #     raise ValueError('Unauthorized plot ID')
        return v

    @validator('numberOfFruits')
    def validate_fruit_count(cls, v):
        if v <= 0:
            raise ValueError('Number of fruits must be positive')
        if v > 10000:
            raise ValueError('Number of fruits exceeds reasonable limit')
        return v

    @validator('harvestDays')
    def validate_harvest_days(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError('Harvest days must be 1, 2, or 3')
        return v

# SQL injection prevention with parameterized queries
async def get_forms_by_tree_safe(tree_id: str) -> List[dict]:
    """
    Safe database query with parameterized statements.
    Prevents SQL injection attacks.
    """
    query = """
        SELECT f.*,
               COALESCE(
                   json_agg(
                       json_build_object(
                           'id', i.id,
                           'url', i.url,
                           'filename', i.filename,
                           'uploaded_at', i.uploaded_at
                       ) ORDER BY i.image_index
                   ) FILTER (WHERE i.id IS NOT NULL),
                   '[]'::json
               ) as images
        FROM recon_forms f
        LEFT JOIN images i ON f.id = i.form_id AND i.form_type = 'recon'
        WHERE f.tree_id = $1
        GROUP BY f.id
        ORDER BY f.created_at DESC
    """

    async with get_db_connection() as conn:
        rows = await conn.fetch(query, tree_id)
        return [dict(row) for row in rows]

# File upload security
async def validate_uploaded_file(file_content: bytes,
                                filename: str,
                                max_size_mb: int = 10) -> bool:
    """
    Validates uploaded files for security threats.
    """
    # Check file size
    if len(file_content) > max_size_mb * 1024 * 1024:
        raise ValueError(f"File too large (max {max_size_mb}MB)")

    # Check file type by content (not just extension)
    file_type = magic.from_buffer(file_content, mime=True)
    allowed_types = [
        'image/jpeg', 'image/png', 'image/webp', 'image/bmp'
    ]

    if file_type not in allowed_types:
        raise ValueError(f"Invalid file type: {file_type}")

    # Check for malicious content
    if contains_malicious_content(file_content):
        raise ValueError("File contains potentially malicious content")

    # Validate image integrity
    try:
        with Image.open(io.BytesIO(file_content)) as img:
            img.verify()  # Verify image integrity
    except Exception:
        raise ValueError("Invalid or corrupted image file")

    return True

def contains_malicious_content(file_content: bytes) -> bool:
    """
    Scans file content for malicious patterns.
    """
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
```

### Rate Limiting & DDoS Protection

```python
# Advanced rate limiting configuration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Create rate limiter with Redis backend for distributed rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",
    strategy="sliding_window"
)

# Different rate limits for different operations
class RateLimits:
    FORM_SUBMISSION = "20/minute"      # Form submissions
    IMAGE_UPLOAD = "50/minute"         # Image uploads
    DATA_QUERY = "200/minute"          # Data queries
    BULK_OPERATIONS = "5/minute"       # Bulk operations
    AUTH_ATTEMPTS = "10/minute"        # Authentication attempts

# Apply rate limiting to endpoints
@app.post("/api/forms")
@limiter.limit(RateLimits.FORM_SUBMISSION)
async def create_form(request: Request, form_data: ReconFormRequest):
    pass

@app.post("/api/image-list")
@limiter.limit(RateLimits.IMAGE_UPLOAD)
async def upload_images(request: Request, image_data: ImageListRequest):
    pass

# Custom rate limit handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    response = JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Rate limit: {exc.detail}",
            "retry_after": exc.retry_after
        }
    )
    response.headers["Retry-After"] = str(exc.retry_after)
    return response
```

---

## Performance Considerations

### Database Optimization

#### **Indexing Strategy**
```sql
-- Primary indexes for fast lookups
CREATE INDEX CONCURRENTLY idx_recon_forms_tree_id ON recon_forms(tree_id);
CREATE INDEX CONCURRENTLY idx_recon_forms_plot_id ON recon_forms(plot_id);
CREATE INDEX CONCURRENTLY idx_recon_forms_created_at ON recon_forms(created_at DESC);

-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY idx_recon_forms_tree_plot_date
    ON recon_forms(tree_id, plot_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_recon_forms_sync_status
    ON recon_forms(is_synced, sync_timestamp)
    WHERE is_synced = false;

-- Partial indexes for specific conditions
CREATE INDEX CONCURRENTLY idx_unsynced_forms
    ON recon_forms(created_at DESC)
    WHERE is_synced = false;

-- Expression indexes for search functionality
CREATE INDEX CONCURRENTLY idx_recon_forms_tree_search
    ON recon_forms(UPPER(tree_id));

-- Spatial indexes for location queries (requires PostGIS)
CREATE INDEX CONCURRENTLY idx_tree_locations_spatial
    ON tree_locations USING GIST(geom);
```

#### **Query Optimization**
```sql
-- Optimized query for dashboard analytics
WITH form_stats AS (
    SELECT
        COUNT(*) as total_forms,
        COUNT(DISTINCT tree_id) as unique_trees,
        COUNT(DISTINCT plot_id) as unique_plots,
        AVG(number_of_fruits) as avg_fruits,
        COUNT(*) FILTER (WHERE is_synced = false) as unsynced_count
    FROM recon_forms
    WHERE created_at >= $1 -- date filter parameter
),
image_stats AS (
    SELECT COUNT(*) as total_images
    FROM images
    WHERE form_type = 'recon'
      AND uploaded_at >= $1
),
daily_trends AS (
    SELECT
        DATE_TRUNC('day', TO_TIMESTAMP(created_at / 1000)) as date,
        COUNT(*) as form_count,
        SUM(number_of_fruits) as fruit_count
    FROM recon_forms
    WHERE created_at >= $1
    GROUP BY DATE_TRUNC('day', TO_TIMESTAMP(created_at / 1000))
    ORDER BY date DESC
    LIMIT 30
)
SELECT
    json_build_object(
        'summary', (SELECT row_to_json(form_stats) FROM form_stats),
        'images', (SELECT row_to_json(image_stats) FROM image_stats),
        'trends', (SELECT json_agg(row_to_json(daily_trends)) FROM daily_trends)
    ) as dashboard_data;
```

### Caching Strategy

#### **Redis Caching Implementation**
```python
import redis
import json
from typing import Optional, Dict, Any

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour default TTL

    async def get_dashboard_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves dashboard data from cache.
        """
        cached_data = await self.redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        return None

    async def cache_dashboard_data(self, cache_key: str,
                                 data: Dict[str, Any],
                                 ttl: int = None) -> None:
        """
        Caches dashboard data with TTL.
        """
        ttl = ttl or self.default_ttl
        await self.redis_client.setex(
            cache_key,
            ttl,
            json.dumps(data, default=str)
        )

    async def invalidate_cache_pattern(self, pattern: str) -> None:
        """
        Invalidates cache entries matching pattern.
        """
        keys = await self.redis_client.keys(pattern)
        if keys:
            await self.redis_client.delete(*keys)

# Cache implementation in API endpoints
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(plot_id: Optional[str] = None):
    cache_key = f"dashboard:{plot_id or 'all'}:{datetime.now().strftime('%Y-%m-%d-%H')}"

    # Try cache first
    cached_data = await cache_manager.get_dashboard_data(cache_key)
    if cached_data:
        return cached_data

    # Generate fresh data
    dashboard_data = await generate_dashboard_data(plot_id)

    # Cache for 1 hour
    await cache_manager.cache_dashboard_data(cache_key, dashboard_data, 3600)

    return dashboard_data

# Cache invalidation on data updates
@app.post("/api/forms")
async def create_form(form_data: ReconFormRequest):
    # Create form...
    result = await create_recon_form(form_data)

    # Invalidate related caches
    await cache_manager.invalidate_cache_pattern("dashboard:*")
    await cache_manager.invalidate_cache_pattern(f"forms:{form_data.treeId}:*")

    return result
```

### Image Processing & CDN

#### **Image Optimization Pipeline**
```python
from PIL import Image
import io
from typing import Dict, Tuple

class ImageProcessor:
    def __init__(self):
        self.max_dimensions = (1920, 1920)  # Max resolution
        self.quality_settings = {
            'thumbnail': {'size': (300, 300), 'quality': 80},
            'medium': {'size': (800, 800), 'quality': 85},
            'full': {'size': (1920, 1920), 'quality': 90}
        }

    async def process_image(self, image_bytes: bytes,
                          filename: str) -> Dict[str, bytes]:
        """
        Processes image into multiple sizes for different use cases.
        """
        original_image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if original_image.mode in ('RGBA', 'P'):
            original_image = original_image.convert('RGB')

        processed_images = {}

        for size_name, settings in self.quality_settings.items():
            # Resize image maintaining aspect ratio
            resized_image = original_image.copy()
            resized_image.thumbnail(settings['size'], Image.Resampling.LANCZOS)

            # Save to bytes
            output_buffer = io.BytesIO()
            resized_image.save(
                output_buffer,
                format='JPEG',
                quality=settings['quality'],
                optimize=True
            )

            processed_images[size_name] = output_buffer.getvalue()

        return processed_images

    async def extract_metadata(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts comprehensive metadata from image.
        """
        image = Image.open(io.BytesIO(image_bytes))

        metadata = {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode,
            'file_size': len(image_bytes)
        }

        # Extract EXIF data
        exif_data = image.getexif()
        if exif_data:
            metadata['exif'] = {
                TAGS.get(tag_id, tag_id): value
                for tag_id, value in exif_data.items()
                if isinstance(value, (str, int, float))
            }

        return metadata

# CDN Integration
class CDNManager:
    def __init__(self, cdn_base_url: str):
        self.cdn_base_url = cdn_base_url

    def generate_image_urls(self, base_filename: str) -> Dict[str, str]:
        """
        Generates CDN URLs for different image sizes.
        """
        return {
            'thumbnail': f"{self.cdn_base_url}/thumbnails/{base_filename}",
            'medium': f"{self.cdn_base_url}/medium/{base_filename}",
            'full': f"{self.cdn_base_url}/full/{base_filename}"
        }

    def get_optimized_url(self, original_url: str,
                         width: int = None,
                         quality: int = None) -> str:
        """
        Generates optimized image URL with transformation parameters.
        """
        params = []
        if width:
            params.append(f"w={width}")
        if quality:
            params.append(f"q={quality}")

        param_string = "&".join(params)
        return f"{original_url}?{param_string}" if params else original_url
```

---

## Monitoring & Logging

### Application Monitoring
```python
import logging
import structlog
from prometheus_client import Counter, Histogram, Gauge
import time
from contextlib import asynccontextmanager

# Metrics collection
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_database_connections', 'Active database connections')
SYNC_OPERATIONS = Counter('sync_operations_total', 'Total sync operations', ['operation', 'status'])

# Structured logging configuration
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Performance monitoring middleware
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()

    # Extract request info
    method = request.method
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code

        # Record metrics
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()

    except Exception as e:
        status_code = 500
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()

        logger.error(
            "Request failed",
            method=method,
            endpoint=endpoint,
            error=str(e),
            exc_info=True
        )
        raise

    finally:
        # Record request duration
        duration = time.time() - start_time
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        logger.info(
            "Request completed",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration=duration
        )

# Database monitoring
@asynccontextmanager
async def monitor_db_operation(operation_name: str):
    """
    Monitors database operations with timing and error tracking.
    """
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()

    try:
        logger.debug("Database operation started", operation=operation_name)
        yield

        duration = time.time() - start_time
        logger.info(
            "Database operation completed",
            operation=operation_name,
            duration=duration
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Database operation failed",
            operation=operation_name,
            duration=duration,
            error=str(e),
            exc_info=True
        )
        raise

    finally:
        ACTIVE_CONNECTIONS.dec()

# Usage example
async def create_recon_form_monitored(form_data: ReconFormRequest):
    async with monitor_db_operation("create_recon_form"):
        # Database operations here
        return await create_recon_form(form_data)
```

---

## Implementation Roadmap

This comprehensive documentation provides a complete blueprint for implementing the backend system to support your Palm Oil Data Collection mobile application. The documentation covers all aspects from basic CRUD operations to advanced features like real-time synchronization, caching, and monitoring.

### Key implementation priorities:

#### **Phase 1: Foundation** (Weeks 1-2)
- Core API endpoints and database schema
- Basic form and image processing
- Authentication and security

#### **Phase 2: Integration** (Weeks 3-4)
- Image processing and blob storage integration
- Data synchronization logic
- Error handling and validation

#### **Phase 3: Enhancement** (Weeks 5-6)
- Advanced features like analytics APIs
- Performance optimization
- Caching implementation

#### **Phase 4: Production** (Weeks 7-8)
- Monitoring and logging
- Load testing and optimization
- Documentation and deployment

The system is designed to scale efficiently while maintaining data integrity and providing excellent user experience for field data collection.

---

## Document Information

**Document Version**: 1.0
**Last Updated**: January 2025
**Author**: System Analysis
**Scope**: Complete backend implementation guide for Palm Oil Data Collection System

This documentation serves as the definitive guide for backend development and should be updated as the system evolves.