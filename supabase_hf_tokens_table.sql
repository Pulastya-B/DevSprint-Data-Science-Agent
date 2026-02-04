-- HuggingFace Tokens Table
-- Run this in your Supabase SQL Editor (supabase.com/dashboard -> SQL Editor)

-- Create the hf_tokens table
CREATE TABLE IF NOT EXISTS hf_tokens (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  huggingface_token TEXT,
  huggingface_username TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE hf_tokens ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for re-running)
DROP POLICY IF EXISTS "Users can view own tokens" ON hf_tokens;
DROP POLICY IF EXISTS "Users can insert own tokens" ON hf_tokens;
DROP POLICY IF EXISTS "Users can update own tokens" ON hf_tokens;
DROP POLICY IF EXISTS "Users can delete own tokens" ON hf_tokens;

-- Policy: Users can only SELECT their own tokens
CREATE POLICY "Users can view own tokens" ON hf_tokens
  FOR SELECT USING (auth.uid() = user_id);

-- Policy: Users can only INSERT their own tokens
CREATE POLICY "Users can insert own tokens" ON hf_tokens
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy: Users can only UPDATE their own tokens
CREATE POLICY "Users can update own tokens" ON hf_tokens
  FOR UPDATE USING (auth.uid() = user_id);

-- Policy: Users can only DELETE their own tokens
CREATE POLICY "Users can delete own tokens" ON hf_tokens
  FOR DELETE USING (auth.uid() = user_id);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_hf_tokens_user_id ON hf_tokens(user_id);

-- Grant access to authenticated users
GRANT ALL ON hf_tokens TO authenticated;

-- Success message
SELECT 'hf_tokens table created successfully!' AS result;
