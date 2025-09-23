# CuliFeed Alpine Docker Images

## Overview

CuliFeed now provides optimized Alpine Linux Docker images alongside the traditional Debian-based images. Alpine images are significantly smaller while maintaining full functionality.

## Image Variants

### Debian (Default)
- **Base**: `python:3.11-slim` (~1.17GB)
- **Tags**: `latest`, `v1.2.3`, `1.2`
- **Use when**: Standard deployment, familiarity with Debian ecosystem

### Alpine (Optimized)
- **Base**: `python:3.11-alpine` (~926MB)  
- **Tags**: `latest-alpine`, `v1.2.3-alpine`, `1.2-alpine`
- **Use when**: Resource constraints, faster deployments, smaller attack surface

## Usage Examples

### Pull Latest Images
```bash
# Debian (default)
docker pull ghcr.io/chiplonton/culifeed:latest

# Alpine (optimized)
docker pull ghcr.io/chiplonton/culifeed:latest-alpine
```

### Version-Specific Images
```bash
# Debian v1.2.3
docker pull ghcr.io/chiplonton/culifeed:v1.2.3

# Alpine v1.2.3  
docker pull ghcr.io/chiplonton/culifeed:v1.2.3-alpine
```

### Docker Compose
```yaml
version: '3.8'
services:
  culifeed:
    # Use Alpine variant
    image: ghcr.io/chiplonton/culifeed:latest-alpine
    environment:
      - TZ=UTC
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.env:/app/.env:ro
```

## Size Comparison

| Variant | Base Image | Final Size | Reduction |
|---------|------------|------------|-----------|
| Debian  | python:3.11-slim | 1.17GB | - |
| Alpine  | python:3.11-alpine | 926MB | **21% smaller** |

## Compatibility

Both variants provide identical functionality:
- ✅ All CuliFeed features work identically
- ✅ Same environment variables and configuration
- ✅ Same volume mounts and networking
- ✅ Same health checks and monitoring

## Migration Guide

### From Debian to Alpine
```bash
# Stop current container
docker stop culifeed

# Pull Alpine image
docker pull ghcr.io/chiplonton/culifeed:latest-alpine

# Update docker-compose.yml or run command
docker run -d \
  --name culifeed \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  -v ./.env:/app/.env:ro \
  ghcr.io/chiplonton/culifeed:latest-alpine
```

### Rollback to Debian
```bash
# Simply use the standard tag
docker pull ghcr.io/chiplonton/culifeed:latest
# Update your deployment to use the non-alpine tag
```

## Production Recommendations

### Use Alpine When:
- ✅ Deploying to resource-constrained environments
- ✅ Network bandwidth is limited
- ✅ Security is a priority (smaller attack surface)
- ✅ Faster container startup is needed

### Use Debian When:
- ✅ Maximum compatibility is required
- ✅ Debugging with familiar tools is important
- ✅ Legacy deployment processes expect Debian

## Available Tags

### Current Release (v1.1.1)
```
ghcr.io/chiplonton/culifeed:v1.1.1        # Debian
ghcr.io/chiplonton/culifeed:v1.1.1-alpine # Alpine
ghcr.io/chiplonton/culifeed:1.1           # Debian
ghcr.io/chiplonton/culifeed:1.1-alpine    # Alpine
ghcr.io/chiplonton/culifeed:latest        # Debian (default)
ghcr.io/chiplonton/culifeed:latest-alpine # Alpine
```

### Future Releases
The tagging pattern will continue for all future releases:
- `vX.Y.Z` - Debian variant (default)
- `vX.Y.Z-alpine` - Alpine variant
- `X.Y-alpine` - Alpine major.minor
- `latest-alpine` - Latest Alpine release

## Build Information

Both images are built simultaneously on GitHub Actions:
- **Platforms**: linux/amd64, linux/arm64
- **Registry**: GitHub Container Registry (ghcr.io)
- **Caching**: Separate cache layers for optimal build performance
- **Release**: Automatic builds triggered by GitHub releases