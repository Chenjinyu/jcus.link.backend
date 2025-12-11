/**
 * Next.js Integration Example for Document Analysis MCP Server
 * 
 * This file shows how to integrate the MCP server with your Next.js application
 */

import { spawn } from 'child_process';

// =============================================================================
// MCP Client Wrapper
// =============================================================================

interface MCPToolCall {
  name: string;
  arguments: Record<string, any>;
}

interface MCPResponse {
  success: boolean;
  data?: any;
  error?: string;
}

/**
 * MCP Client for communicating with the Python MCP server
 */
class DocumentAnalysisMCPClient {
  private process: any;
  private responseHandlers: Map<string, (data: any) => void> = new Map();
  
  constructor() {
    // Spawn the MCP server process
    this.process = spawn('uv', ['run', 'python', 'document_analysis_mcp.py'], {
      cwd: process.env.MCP_SERVER_PATH || './mcp-server'
    });
    
    // Handle server responses
    this.process.stdout.on('data', (data: Buffer) => {
      try {
        const response = JSON.parse(data.toString());
        const handler = this.responseHandlers.get(response.id);
        if (handler) {
          handler(response);
          this.responseHandlers.delete(response.id);
        }
      } catch (error) {
        console.error('Failed to parse MCP response:', error);
      }
    });
    
    this.process.stderr.on('data', (data: Buffer) => {
      console.error('MCP Server Error:', data.toString());
    });
  }
  
  /**
   * Call an MCP tool
   */
  async callTool(toolName: string, args: Record<string, any>): Promise<MCPResponse> {
    return new Promise((resolve, reject) => {
      const id = Math.random().toString(36).substring(7);
      
      const request = {
        jsonrpc: '2.0',
        id,
        method: 'tools/call',
        params: {
          name: toolName,
          arguments: args
        }
      };
      
      // Register response handler
      this.responseHandlers.set(id, (response: any) => {
        if (response.error) {
          reject(new Error(response.error.message));
        } else {
          resolve(response.result);
        }
      });
      
      // Send request to server
      this.process.stdin.write(JSON.stringify(request) + '\n');
      
      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.responseHandlers.has(id)) {
          this.responseHandlers.delete(id);
          reject(new Error('Request timeout'));
        }
      }, 30000);
    });
  }
  
  /**
   * Close the MCP server connection
   */
  close() {
    this.process.kill();
  }
}

// =============================================================================
// Next.js API Route Example
// =============================================================================

/**
 * API Route: /api/analyze-job
 * 
 * Handles job description analysis requests
 */

// app/api/analyze-job/route.ts
import { NextRequest, NextResponse } from 'next/server';

// Singleton MCP client instance
let mcpClient: DocumentAnalysisMCPClient | null = null;

function getMCPClient() {
  if (!mcpClient) {
    mcpClient = new DocumentAnalysisMCPClient();
  }
  return mcpClient;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { jobDescription, profileCollection } = body;
    
    if (!jobDescription) {
      return NextResponse.json(
        { error: 'Job description is required' },
        { status: 400 }
      );
    }
    
    const client = getMCPClient();
    
    // Call the MCP tool
    const result = await client.callTool('analyze_job_description', {
      job_description: jobDescription,
      profile_collection: profileCollection || 'profile',
      top_k_matches: 10,
      embedding_provider: 'ollama',
      embedding_model: 'nomic-embed-text',
      response_format: 'json'
    });
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error analyzing job:', error);
    return NextResponse.json(
      { error: 'Failed to analyze job description' },
      { status: 500 }
    );
  }
}

// =============================================================================
// API Route: /api/upload-document
// =============================================================================

/**
 * Handles document uploads and creates embeddings
 */

// app/api/upload-document/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const fileType = formData.get('fileType') as string;
    const collectionName = formData.get('collectionName') as string || 'default';
    
    if (!file) {
      return NextResponse.json(
        { error: 'File is required' },
        { status: 400 }
      );
    }
    
    const client = getMCPClient();
    
    // Step 1: Extract content from file
    let extractParams: any = {
      file_type: fileType
    };
    
    // Convert file to base64 for PDF/DOCX
    if (fileType === 'pdf' || fileType === 'docx') {
      const bytes = await file.arrayBuffer();
      const base64 = Buffer.from(bytes).toString('base64');
      extractParams.base64_content = base64;
    } else {
      const text = await file.text();
      extractParams.content = text;
    }
    
    const extractResult = await client.callTool('extract_content', extractParams);
    
    if (!extractResult.success) {
      throw new Error(extractResult.error || 'Failed to extract content');
    }
    
    // Step 2: Create embeddings
    const embeddingResult = await client.callTool('create_embeddings', {
      content: extractResult.content,
      collection_name: collectionName,
      chunk_size: 1000,
      chunk_overlap: 200,
      embedding_provider: 'ollama',
      metadata: {
        filename: file.name,
        file_type: fileType,
        upload_date: new Date().toISOString()
      }
    });
    
    return NextResponse.json({
      success: true,
      extracted: extractResult,
      embeddings: embeddingResult
    });
  } catch (error) {
    console.error('Error uploading document:', error);
    return NextResponse.json(
      { error: 'Failed to process document' },
      { status: 500 }
    );
  }
}

// =============================================================================
// React Component Example
// =============================================================================

/**
 * Job Analysis Component
 * 
 * Shows how to use the API from the frontend
 */

// components/JobAnalyzer.tsx
'use client';

import { useState } from 'react';

interface AnalysisResult {
  fit_score: number;
  is_good_fit: boolean;
  matched_requirements: string[];
  missing_requirements: string[];
  relevant_experience: any[];
}

export function JobAnalyzer() {
  const [jobDescription, setJobDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const analyzeJob = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/analyze-job', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobDescription,
          profileCollection: 'profile'
        })
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const data = await response.json();
      setResult(data.analysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Job Description Analyzer</h1>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            Paste Job Description
          </label>
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            className="w-full h-64 p-3 border rounded-lg"
            placeholder="Paste the job description here..."
          />
        </div>
        
        <button
          onClick={analyzeJob}
          disabled={loading || !jobDescription}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-400"
        >
          {loading ? 'Analyzing...' : 'Analyze Fit'}
        </button>
        
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}
        
        {result && (
          <div className="space-y-6 mt-6">
            <div className="p-6 bg-white border rounded-lg shadow">
              <h2 className="text-2xl font-bold mb-4">Analysis Results</h2>
              
              <div className="mb-6">
                <div className="flex items-center gap-4">
                  <div className="text-4xl font-bold">
                    {result.fit_score}%
                  </div>
                  <div className={`px-4 py-2 rounded-full ${
                    result.is_good_fit 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {result.is_good_fit ? '✅ Good Fit' : '⚠️ Skills Gap'}
                  </div>
                </div>
              </div>
              
              {result.matched_requirements.length > 0 && (
                <div className="mb-4">
                  <h3 className="font-semibold text-green-700 mb-2">
                    ✅ Matched Requirements
                  </h3>
                  <ul className="list-disc pl-5 space-y-1">
                    {result.matched_requirements.map((req, i) => (
                      <li key={i} className="text-sm">{req}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {result.missing_requirements.length > 0 && (
                <div className="mb-4">
                  <h3 className="font-semibold text-red-700 mb-2">
                    ❌ Missing Requirements
                  </h3>
                  <ul className="list-disc pl-5 space-y-1">
                    {result.missing_requirements.map((req, i) => (
                      <li key={i} className="text-sm">{req}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {result.relevant_experience.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">
                    📋 Relevant Experience
                  </h3>
                  <div className="space-y-2">
                    {result.relevant_experience.map((exp, i) => (
                      <div key={i} className="p-3 bg-gray-50 rounded">
                        <p className="text-sm">{exp.content}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Document Upload Component
// =============================================================================

/**
 * Component for uploading and processing documents
 */

// components/DocumentUploader.tsx
'use client';

import { useState } from 'react';

export function DocumentUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  
  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Detect file type
    const extension = file.name.split('.').pop()?.toLowerCase();
    const fileTypeMap: Record<string, string> = {
      'pdf': 'pdf',
      'docx': 'docx',
      'doc': 'docx',
      'txt': 'text',
      'md': 'markdown',
      'json': 'json'
    };
    
    formData.append('fileType', fileTypeMap[extension || ''] || 'text');
    formData.append('collectionName', 'profile');
    
    try {
      const response = await fetch('/api/upload-document', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };
  
  return (
    <div className="max-w-2xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-4">Upload Your Resume/Profile</h2>
      
      <div className="space-y-4">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            type="file"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            accept=".pdf,.docx,.doc,.txt,.md,.json"
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer text-blue-600 hover:text-blue-700"
          >
            {file ? file.name : 'Click to upload file'}
          </label>
          <p className="text-sm text-gray-500 mt-2">
            Supports PDF, DOCX, TXT, Markdown, JSON
          </p>
        </div>
        
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="w-full px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-400"
        >
          {uploading ? 'Processing...' : 'Upload and Process'}
        </button>
        
        {result && (
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <h3 className="font-semibold text-green-700 mb-2">Success!</h3>
            <p className="text-sm">
              Created {result.embeddings.num_embeddings} embeddings from{' '}
              {result.embeddings.num_chunks} chunks
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Environment Variables
// =============================================================================

/**
 * Add to your .env.local:
 * 
 * MCP_SERVER_PATH=/path/to/mcp-server
 * 
 * Optional (if using OpenAI):
 * OPENAI_API_KEY=your_key_here
 */