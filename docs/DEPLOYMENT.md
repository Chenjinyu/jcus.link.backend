# Deployment Guide - Document Analysis MCP Server

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Web UI     │  │  Mobile App  │  │   Desktop    │           │
│  │  (Next.js)   │  │   (React)    │  │    (Tauri)   │           │
│  └──────┬───────┘  └───────┬──────┘  └────────┬─────┘           │
│         │                  │                  │                 │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          │   HTTP/WebSocket │                  │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼──────────────────┐
│                     Application Layer                            │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              Next.js API Routes / Express                 │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │   │
│  │  │  /api/upload│  │ /api/analyze │  │  /api/query   │     │   │
│  │  └──────┬──────┘  └───────┬──────┘  └───────┬───────┘     │   │
│  └─────────┼─────────────────┼─────────────────┼─────────────┘   │
│            │                 │                 │                 │
└────────────┼─────────────────┼─────────────────┼─────────────────┘
             │                 │                 │
             │  MCP Protocol   │                 │
             │  (stdio/HTTP)   │                 │
┌────────────▼─────────────────▼─────────────────▼─────────────────┐
│                        MCP Server Layer                          │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │         Document Analysis MCP Server (FastMCP)           │    │
│  │                                                          │    │
│  │  ┌────────────────┐  ┌─────────────────┐                 │    │
│  │  │ extract_content │  │create_embeddings│                │    │
│  │  └────────┬───────┘  └────────┬────────┘                 │    │
│  │                                │                         │    │
│  │  ┌─────────────────┐  ┌───────▼────────────┐             │    │
│  │  │ query_embeddings│  │analyze_job_description│          │    │
│  │  └────────┬────────┘  └────────┬────────────┘            │    │
│  └───────────┼──────────────────────┼───────────────────────│    │
│              │                      │                       │    │
└──────────────┼──────────────────────┼────────────────────────────┘
               │                      │
      ┌────────┴────────────┐      ┌──────▼─────────┐
      │                     │      │                │
┌─────▼────────────-┐  ┌────▼──────▼───┐  ┌─────────▼────────┐
│  Document         │  │   Embedding   │  │  Vector Database │
│  Loaders          │  │   Providers   │  │    (ChromaDB)    │
│                   │  │               │  │                  │
│ • PyPDF           │  │ • Ollama      │  │ • Collections    │
│ • Docx2txt        │  │ • OpenAI      │  │ • Similarity     │
│ • BeautifulSoup.  │  │ • Custom      │  │ • Metadata       │
│ • Markdown        │  │               │  │ • Persistence    │
└───────────────────┘  └───────────────┘  └──────────────────┘
```

## Deployment Options

### Option 1: Local Development

**Best for**: Development, testing, small-scale use

```bash
# 1. Clone and setup
git clone <your-repo>
cd document-analysis-mcp
./setup.sh

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start Ollama (for local embeddings)
ollama serve

# 4. Run MCP server
uv run python document_analysis_mcp.py
```

**Pros**: 
- Free (no API costs)
- Full privacy (all data local)
- Fast iteration

**Cons**:
- Requires local Ollama installation
- Limited to single machine
- Manual scaling

### Option 2: Docker Deployment

**Best for**: Production, consistent environments, easy scaling

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY document_analysis_mcp.py .
COPY .env .

# Install dependencies
RUN uv sync

# Expose port (if using HTTP transport)
EXPOSE 8000

# Run server
CMD ["uv", "run", "python", "document_analysis_mcp.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMA_PERSIST_DIR=/data/chroma
    volumes:
      - ./chroma_db:/data/chroma
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    command: serve

volumes:
  ollama_data:
```

**Deploy**:
```bash
docker-compose up -d
```

**Pros**:
- Consistent environment
- Easy deployment
- Scalable with orchestration
- Isolated dependencies

**Cons**:
- Requires Docker knowledge
- More complex setup

### Option 3: Cloud Deployment (AWS)

**Best for**: Production, high availability, auto-scaling

#### Architecture on AWS

```
┌────────────────────────────────────────────────┐
│              Route 53 (DNS)                    │
└─────────────────┬──────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────┐
│       CloudFront (CDN) + ACM (SSL)             │
└─────────────────┬──────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────┐
│         Application Load Balancer               │
└─────────────────┬──────────────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
┌────────▼────────┐  ┌────▼─────────────────┐
│   ECS/Fargate   │  │    Lambda Function   │
│   (MCP Server)  │  │   (Alternative)      │
│                 │  │                      │
│ • Auto Scaling  │  │ • Serverless         │
│ • Health Checks │  │ • Event-driven       │
└────────┬────────┘  └──────────────────────┘
         │
         └──────┬───────────────┐
                │               │
     ┌──────────▼──────┐  ┌────▼──────────┐
     │  EFS (Shared)   │  │  S3 (Storage) │
     │  ChromaDB Data  │  │  Documents    │
     └─────────────────┘  └───────────────┘
```

#### Deployment Steps

**1. Setup ECS Cluster**

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name mcp-server-cluster

# Create ECR repository
aws ecr create-repository --repository-name document-analysis-mcp

# Build and push image
docker build -t document-analysis-mcp .
docker tag document-analysis-mcp:latest <account-id>.dkr.ecr.<region>.amazonaws.com/document-analysis-mcp:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/document-analysis-mcp:latest
```

**2. Create EFS for ChromaDB**

```bash
# Create EFS file system
aws efs create-file-system --tags Key=Name,Value=mcp-chroma-db

# Create mount targets in each AZ
aws efs create-mount-target \
  --file-system-id <fs-id> \
  --subnet-id <subnet-id> \
  --security-groups <sg-id>
```

**3. Deploy with Terraform**

```hcl
# main.tf
resource "aws_ecs_task_definition" "mcp_server" {
  family                   = "document-analysis-mcp"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  
  container_definitions = jsonencode([{
    name  = "mcp-server"
    image = "${aws_ecr_repository.mcp.repository_url}:latest"
    
    environment = [
      { name = "OLLAMA_BASE_URL", value = var.ollama_url },
      { name = "OPENAI_API_KEY", value = var.openai_api_key },
    ]
    
    mountPoints = [{
      sourceVolume  = "chroma-data"
      containerPath = "/app/chroma_db"
    }]
    
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
  }])
  
  volume {
    name = "chroma-data"
    
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.chroma.id
      root_directory = "/"
    }
  }
}

resource "aws_ecs_service" "mcp_server" {
  name            = "mcp-server-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mcp_server.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.mcp_server.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.mcp_server.arn
    container_name   = "mcp-server"
    container_port   = 8000
  }
}
```

**Deploy**:
```bash
terraform init
terraform plan
terraform apply
```

**Pros**:
- Highly scalable
- Managed infrastructure
- High availability
- Auto-scaling
- Monitoring included

**Cons**:
- More complex
- Higher cost
- Requires AWS expertise

### Option 4: Kubernetes Deployment

**Best for**: Multi-cloud, large scale, complex deployments

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: document-analysis-mcp:latest
        ports:
        - containerPort: 8000
        env:
        - name: OLLAMA_BASE_URL
          value: "http://ollama-service:11434"
        - name: CHROMA_PERSIST_DIR
          value: "/data/chroma"
        volumeMounts:
        - name: chroma-storage
          mountPath: /data/chroma
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: chroma-storage
        persistentVolumeClaim:
          claimName: chroma-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chroma-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
```

**Deploy**:
```bash
kubectl apply -f k8s/
```

## Environment Configuration

### Development
```env
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
DEFAULT_EMBEDDING_PROVIDER=ollama
CHROMA_PERSIST_DIR=./chroma_db
```

### Staging
```env
OLLAMA_BASE_URL=http://ollama-staging.internal:11434
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
DEFAULT_EMBEDDING_PROVIDER=ollama
CHROMA_PERSIST_DIR=/mnt/efs/chroma_db
```

### Production
```env
# Use OpenAI for better quality
DEFAULT_EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-proj-xxxxx
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=/mnt/efs/chroma_db

# Or use Ollama for privacy
OLLAMA_BASE_URL=http://ollama-prod.internal:11434
DEFAULT_EMBEDDING_PROVIDER=ollama
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
```

## Monitoring and Observability

### Metrics to Track

1. **Performance Metrics**
   - Request latency
   - Embedding creation time
   - Query response time
   - Document processing time

2. **Resource Metrics**
   - CPU usage
   - Memory usage
   - Disk usage (ChromaDB)
   - Network throughput

3. **Business Metrics**
   - Documents processed
   - Embeddings created
   - Queries executed
   - User sessions

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards

  loki:
    image: grafana/loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml

volumes:
  prometheus_data:
  grafana_data:
```

## Security Best Practices

1. **API Keys**
   - Store in AWS Secrets Manager or HashiCorp Vault
   - Rotate regularly
   - Never commit to git

2. **Network Security**
   - Use VPC for AWS deployments
   - Enable security groups/firewalls
   - TLS/SSL for all connections

3. **Access Control**
   - Implement authentication
   - Use API keys for programmatic access
   - Rate limiting

4. **Data Privacy**
   - Encrypt data at rest (EFS encryption)
   - Encrypt data in transit (TLS)
   - Consider using local Ollama for sensitive data

## Scaling Considerations

### Vertical Scaling
- Increase container CPU/memory
- Use larger EC2/Fargate instances
- More powerful GPU for embeddings

### Horizontal Scaling
- Multiple MCP server instances
- Load balancer distribution
- Shared ChromaDB via EFS/network storage

### Database Scaling
- Partition ChromaDB by collection
- Use multiple ChromaDB instances
- Consider managed vector DB (Pinecone, Weaviate)

## Backup and Disaster Recovery

```bash
# Backup ChromaDB
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_db/

# Upload to S3
aws s3 cp chroma_backup_*.tar.gz s3://my-backups/chroma/

# Automate with cron
0 2 * * * /path/to/backup.sh
```

## Cost Optimization

### Using Ollama (Local)
- **Pros**: Free, no API costs
- **Cons**: Requires GPU hardware
- **Best for**: Privacy-sensitive, high-volume

### Using OpenAI
- **Costs**: ~$0.00002 per 1K tokens
- **Example**: 1M documents = ~$20-50/month
- **Best for**: Best quality, low maintenance

### Hybrid Approach
- OpenAI for new embeddings (quality)
- Cache results in ChromaDB
- Reduce repeated API calls

## Troubleshooting

### Common Issues

**1. Ollama Connection Refused**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

**2. ChromaDB Permission Denied**
```bash
chmod -R 755 chroma_db/
chown -R $USER:$USER chroma_db/
```

**3. Out of Memory**
```bash
# Reduce chunk size
chunk_size=500  # instead of 1000

# Reduce batch size
# Process documents one at a time
```

**4. Slow Query Performance**
```bash
# Add indexes to ChromaDB
# Reduce top_k parameter
# Use metadata filtering
```

## Support and Maintenance

- Regular updates: `uv sync --upgrade`
- Monitor logs: `docker logs -f mcp-server`
- Health checks: `/health` endpoint
- Backup schedule: Daily at 2 AM