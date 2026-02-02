# File Storage Architecture - Implementation Guide

## Overview

This document outlines the complete file storage architecture for persisting user files (plots, CSVs, reports, models) across sessions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STORAGE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Frontend (React)                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  • PlotRenderer.tsx - Renders Plotly charts from JSON           │   │
│   │  • Assets panel - Shows user files from Supabase                │   │
│   │  • Download buttons - Uses presigned R2 URLs                    │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   Backend (FastAPI)                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  /api/files - List user files                                   │   │
│   │  /api/files/{id} - Get file with download URL                   │   │
│   │  /api/files/stats/{user_id} - Storage statistics                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│              ┌───────────────┴───────────────┐                          │
│              ▼                               ▼                          │
│   Supabase (Metadata)              Cloudflare R2 (Files)                │
│   ┌─────────────────┐              ┌─────────────────────┐              │
│   │  user_files     │              │  /users/{user_id}/  │              │
│   │  - id           │  ──────────► │    /plots/*.json.gz │              │
│   │  - user_id      │              │    /data/*.csv.gz   │              │
│   │  - r2_key       │              │    /reports/*.html  │              │
│   │  - expires_at   │              │    /models/*.pkl.gz │              │
│   └─────────────────┘              └─────────────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Setup Steps

### 1. Cloudflare R2 Setup

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Navigate to R2 → Create Bucket → Name it `ds-agent-files`
3. Go to R2 → Manage R2 API Tokens → Create API Token
4. Note down:
   - Account ID (from URL or overview page)
   - Access Key ID
   - Secret Access Key

### 2. Environment Variables

Add to your `.env` file:

```bash
# Cloudflare R2
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=ds-agent-files
R2_PUBLIC_URL=  # Optional: custom domain

# Supabase (existing)
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
```

### 3. Supabase Table

Run this SQL in Supabase SQL Editor:

```sql
CREATE TABLE user_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id TEXT,
    file_type TEXT NOT NULL CHECK (file_type IN ('plot', 'csv', 'report', 'model')),
    file_name TEXT NOT NULL,
    r2_key TEXT NOT NULL UNIQUE,
    size_bytes BIGINT,
    mime_type TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days'),
    is_deleted BOOLEAN DEFAULT FALSE
);

-- Indexes
CREATE INDEX idx_user_files_user_id ON user_files(user_id);
CREATE INDEX idx_user_files_session ON user_files(session_id);
CREATE INDEX idx_user_files_expires ON user_files(expires_at) WHERE NOT is_deleted;

-- RLS Policies
ALTER TABLE user_files ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own files" ON user_files
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own files" ON user_files
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own files" ON user_files
    FOR DELETE USING (auth.uid() = user_id);
```

### 4. Python Dependencies

Add to `requirements.txt`:

```
boto3>=1.28.0
```

## Usage in Orchestrator

When generating files in the orchestrator, save them to R2:

```python
from src.storage.r2_storage import store_plotly_figure, store_dataframe_csv
from src.storage.user_files_service import get_files_service, FileType

# Store a Plotly figure
def save_plot(user_id: str, session_id: str, fig, plot_name: str):
    r2_key, size = store_plotly_figure(user_id, fig, plot_name)
    
    # Record in Supabase
    files_service = get_files_service()
    files_service.create_file_record(
        user_id=user_id,
        file_type=FileType.PLOT,
        file_name=plot_name,
        r2_key=r2_key,
        size_bytes=size,
        session_id=session_id,
        mime_type='application/json',
        metadata={'plot_type': 'plotly'}
    )
    
    return r2_key

# Store a CSV
def save_csv(user_id: str, session_id: str, df, filename: str):
    r2_key, compressed_size, original_size = store_dataframe_csv(
        user_id, df, filename, "Processed dataset"
    )
    
    files_service = get_files_service()
    files_service.create_file_record(
        user_id=user_id,
        file_type=FileType.CSV,
        file_name=filename,
        r2_key=r2_key,
        size_bytes=compressed_size,
        session_id=session_id,
        mime_type='text/csv',
        metadata={
            'original_size': original_size,
            'compression_ratio': f"{(1 - compressed_size/original_size)*100:.1f}%"
        }
    )
    
    return r2_key
```

## Storage Efficiency

### Plot Storage (Before vs After)

| Format | Size | Load Time |
|--------|------|-----------|
| Plotly HTML | 200KB - 2MB | 2-5 seconds |
| Plotly JSON (gzip) | 5KB - 20KB | <0.5 seconds |

**95% reduction in storage!**

### CSV Compression

| Original Size | Compressed (gzip) | Ratio |
|---------------|-------------------|-------|
| 10MB | 1-2MB | 80-90% |
| 100MB | 10-20MB | 80-90% |
| 1GB | 100-200MB | 80-90% |

## Cleanup Strategy

### Automatic Expiration

Files expire after 7 days by default. Run this cleanup job daily:

```python
from src.storage.r2_storage import get_r2_service
from src.storage.user_files_service import get_files_service

def cleanup_expired_files():
    files_service = get_files_service()
    r2_service = get_r2_service()
    
    # Get expired files from Supabase
    expired = files_service.get_expired_files()
    
    for file in expired:
        # Delete from R2
        r2_service.delete_file(file.r2_key)
        # Delete from Supabase
        files_service.hard_delete_file(file.id)
    
    return len(expired)
```

### User Download Prompt

When files are about to expire (1 day left), show a notification:

```typescript
// Frontend
const expiringFiles = files.filter(f => 
  new Date(f.expires_at) < new Date(Date.now() + 24 * 60 * 60 * 1000)
);

if (expiringFiles.length > 0) {
  showNotification(
    `${expiringFiles.length} files expiring soon! Download them now.`
  );
}
```

## Cost Estimates

### Cloudflare R2 (10GB free, then $0.015/GB)

| Users | Files/User | Avg Size | Total Storage | Monthly Cost |
|-------|------------|----------|---------------|--------------|
| 100 | 50 | 500KB | 2.5GB | FREE |
| 1,000 | 50 | 500KB | 25GB | $0.23 |
| 10,000 | 50 | 500KB | 250GB | $3.60 |

**Zero egress fees = users can download unlimited files for free!**

## Next Steps

1. ✅ Created R2StorageService (`src/storage/r2_storage.py`)
2. ✅ Created UserFilesService (`src/storage/user_files_service.py`)
3. ✅ Added API endpoints to `app.py`
4. ✅ Created PlotRenderer component
5. ⏳ TODO: Integrate with orchestrator to save files during workflow
6. ⏳ TODO: Update frontend Assets panel to fetch from API
7. ⏳ TODO: Add expiration notifications
8. ⏳ TODO: Set up daily cleanup cron job
