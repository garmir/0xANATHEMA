# Task Master AI - Production Docker Image
# Multi-stage build for optimized container size

# Build stage
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    make \
    g++ \
    git \
    bash \
    curl

# Copy package files
COPY package*.json ./
COPY requirements.txt ./

# Install Node.js dependencies
RUN npm ci --only=production && npm cache clean --force

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build the application (if needed)
RUN npm run build 2>/dev/null || echo "No build script found"

# Remove development files
RUN rm -rf \
    .git \
    .github \
    docs \
    tests \
    *.md \
    .gitignore \
    .dockerignore \
    Dockerfile*

# Production stage
FROM node:20-alpine AS production

# Install runtime dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    bash \
    curl \
    git \
    openssh-client \
    ca-certificates \
    tzdata \
    tini

# Create non-root user
RUN addgroup -g 1001 -S taskmaster && \
    adduser -u 1001 -S taskmaster -G taskmaster

# Set working directory
WORKDIR /app

# Copy application from builder stage
COPY --from=builder --chown=taskmaster:taskmaster /app ./

# Install global task-master-ai package
RUN npm install -g task-master-ai@latest

# Create necessary directories
RUN mkdir -p \
    /.taskmaster \
    /workspace \
    /cache \
    /logs \
    /backup \
    /scripts && \
    chown -R taskmaster:taskmaster \
    /.taskmaster \
    /workspace \
    /cache \
    /logs \
    /backup \
    /scripts

# Copy startup scripts
COPY --chown=taskmaster:taskmaster scripts/docker/ /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh

# Health check script
COPY --chown=taskmaster:taskmaster <<EOF /usr/local/bin/healthcheck.sh
#!/bin/bash
set -e

# Check if task-master is responding
if ! curl -f -s http://localhost:8080/health > /dev/null; then
    echo "Health check failed: API not responding"
    exit 1
fi

# Check task-master status
if ! task-master status --quiet; then
    echo "Health check failed: Task master status check failed"
    exit 1
fi

echo "Health check passed"
exit 0
EOF

RUN chmod +x /usr/local/bin/healthcheck.sh

# Set up environment
ENV NODE_ENV=production \
    LOG_LEVEL=info \
    PORT=8080 \
    METRICS_PORT=9090 \
    PATH="/usr/local/bin:$PATH"

# Switch to non-root user
USER taskmaster

# Expose ports
EXPOSE 8080 9090

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Set entrypoint
ENTRYPOINT ["/sbin/tini", "--"]

# Default command
CMD ["/usr/local/bin/start-task-master.sh"]

# Labels for metadata
LABEL \
    org.opencontainers.image.title="Task Master AI" \
    org.opencontainers.image.description="Autonomous AI task execution system with research integration" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.vendor="Task Master AI" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.source="https://github.com/your-org/task-master-ai" \
    org.opencontainers.image.documentation="https://docs.task-master-ai.com" \
    maintainer="devops@task-master-ai.com"