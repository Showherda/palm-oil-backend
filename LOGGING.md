# Logging Documentation

## Overview

The Palm Oil Backend API now includes comprehensive logging capabilities with structured output, log rotation, and contextual information tracking. This document describes the logging implementation and how to use it.

## Features

### 1. **Centralized Logging Configuration**
- Located in `logging_config.py`
- Configurable log levels, formats, and output destinations
- Support for both JSON and text-based log formats
- Automatic log rotation to prevent disk space issues

### 2. **Multiple Log Files**
The system generates four separate log files in the `logs/` directory:

- **`app.log`** - All application logs (DEBUG level and above)
- **`error.log`** - Only errors and critical issues
- **`access.log`** - HTTP request/response logs
- **`database.log`** - Database operation logs

### 3. **Structured Logging with Context**
All log entries include:
- Timestamp (ISO 8601 format)
- Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Module and function name
- Line number
- Contextual metadata (request IDs, user IDs, etc.)

### 4. **Log Rotation**
- Each log file rotates when it reaches 10MB
- Keeps 5 backup files (10MB × 5 = 50MB per log type)
- Automatic cleanup of old logs

### 5. **Performance Monitoring**
- Request duration tracking
- Slow request detection (> 5 seconds)
- Success rate metrics for batch operations

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Logging configuration
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
ENABLE_JSON_LOGS=true            # true for JSON format, false for text
LOG_DIR=logs                      # Directory for log files
```

### Log Levels

- **DEBUG** - Detailed diagnostic information
- **INFO** - General informational messages
- **WARNING** - Warning messages (non-critical issues)
- **ERROR** - Error messages (failures)
- **CRITICAL** - Critical issues requiring immediate attention

## Log Format

### JSON Format (Default)
```json
{
  "timestamp": "2025-10-01T14:30:45.123456",
  "level": "INFO",
  "logger": "__main__",
  "message": "Recon form created successfully",
  "module": "index",
  "function": "create_recon_form",
  "line": 790,
  "form_id": 123,
  "tree_id": "TREE-001",
  "plot_id": "PLOT-A"
}
```

### Text Format
```
2025-10-01 14:30:45 - __main__ - INFO - index:create_recon_form:790 - Recon form created successfully
```

## What Gets Logged

### 1. **Application Lifecycle**
- Startup/shutdown events
- Database initialization
- Connection pool creation

### 2. **HTTP Requests**
- All incoming requests with:
  - Request ID (for tracing)
  - Method and endpoint
  - Client IP address
  - User agent
  - Query parameters
- Response status codes
- Request duration
- Slow request warnings

### 3. **Database Operations**
- Connection pool management
- Table creation and initialization
- Query execution for critical operations
- Transaction success/failure

### 4. **Business Logic**
- Form creation (recon forms, harvester proofs, tree locations)
- Image processing and validation
- Duplicate detection
- Data validation failures

### 5. **Security Events**
- Invalid API key attempts
- Failed authentication
- URL validation failures
- File validation failures

### 6. **Errors and Exceptions**
- All exceptions with full stack traces
- Validation errors with details
- Database errors
- External service failures (image downloads, etc.)

## Contextual Information

Logs include relevant context based on the operation:

### Form Operations
- `form_id` - Database ID of the form
- `tree_id` - Tree identifier
- `plot_id` - Plot identifier
- `client_id` - Client application ID

### HTTP Requests
- `request_id` - Unique request identifier
- `method` - HTTP method (GET, POST, etc.)
- `endpoint` - API endpoint path
- `status_code` - HTTP response code
- `duration_ms` - Request processing time

### Image Processing
- `image_count` - Number of images in batch
- `processed` - Successfully processed images
- `errors` - Number of errors
- `success_rate` - Processing success percentage
- `checksum` - Image checksum for deduplication

### Database Operations
- `form_id` - Related form ID
- `image_timestamp` - Extracted timestamp from EXIF
- `query_params` - SQL query parameters (when relevant)

## Usage Examples

### Reading Logs

**View recent application logs:**
```bash
tail -f logs/app.log
```

**View only errors:**
```bash
tail -f logs/error.log
```

**View access logs:**
```bash
tail -f logs/access.log
```

**Search for specific tree ID:**
```bash
grep "TREE-001" logs/app.log
```

**Parse JSON logs with jq:**
```bash
cat logs/app.log | jq 'select(.tree_id == "TREE-001")'
```

**Find slow requests:**
```bash
cat logs/access.log | jq 'select(.duration_ms > 5000)'
```

### Monitoring

**Count errors in last hour:**
```bash
cat logs/error.log | jq 'select(.timestamp > "'$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)'")'  | wc -l
```

**Average request duration:**
```bash
cat logs/access.log | jq '.duration_ms' | awk '{sum+=$1; count++} END {print sum/count}'
```

**Most common errors:**
```bash
cat logs/error.log | jq -r '.message' | sort | uniq -c | sort -rn | head -10
```

## Best Practices

### 1. **Log Rotation Management**
- Monitor disk space regularly
- Adjust `max_bytes` and `backup_count` based on traffic volume
- Archive old logs if needed for compliance

### 2. **Sensitive Data**
- API keys are never logged
- Passwords are never logged
- PII is minimized in logs
- Use request IDs to trace operations without exposing user data

### 3. **Performance**
- JSON logging has minimal overhead
- Async operations don't block on logging
- Use appropriate log levels (avoid DEBUG in production)

### 4. **Alerting**
Set up alerts for:
- ERROR and CRITICAL level logs
- High error rates
- Slow request warnings
- Authentication failures

### 5. **Log Retention**
- Production: Keep logs for 30-90 days
- Development: Keep logs for 7-14 days
- Compliance: Check regulatory requirements

## Integration with Log Aggregation

### ELK Stack (Elasticsearch, Logstash, Kibana)
```yaml
# logstash.conf
input {
  file {
    path => "/path/to/logs/*.log"
    codec => "json"
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "palm-oil-api-%{+YYYY.MM.dd}"
  }
}
```

### CloudWatch (AWS)
Use the CloudWatch Logs agent to stream logs to AWS.

### Datadog
Install the Datadog agent and point it to the logs directory.

### Splunk
Configure Splunk Universal Forwarder to collect logs.

## Troubleshooting

### Logs Not Appearing
1. Check log directory permissions
2. Verify `LOG_DIR` environment variable
3. Check disk space
4. Ensure application has write permissions

### Large Log Files
1. Reduce log level to WARNING or ERROR
2. Decrease `backup_count`
3. Implement log archiving
4. Use log aggregation services

### Performance Impact
1. Use INFO or WARNING level in production
2. Enable JSON logs only if needed
3. Ensure adequate disk I/O
4. Consider async log handlers for high-volume scenarios

## Development vs Production

### Development Settings
```bash
LOG_LEVEL=DEBUG
ENABLE_JSON_LOGS=false
```

### Production Settings
```bash
LOG_LEVEL=INFO
ENABLE_JSON_LOGS=true
```

## Summary

The logging system provides:
- ✅ Comprehensive request/response tracking
- ✅ Structured logs with contextual information
- ✅ Automatic log rotation
- ✅ Performance monitoring
- ✅ Security event tracking
- ✅ Database operation logging
- ✅ Error tracking with stack traces
- ✅ Easy integration with log aggregation tools

All logs are timestamped, categorized by severity, and include relevant context for debugging and monitoring.
