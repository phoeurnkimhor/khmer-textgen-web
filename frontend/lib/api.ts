// API Base Configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL

// API Response Types
export interface GenerateResponse {
  generated_text: string
}

export interface TrainingResponse {
  message: string
  test_perplexity: number
  test_accuracy: number
  model_path: string
  public_url: string
}

// API Request Types
export interface GenerateRequest {
  text: string
  length?: number
  seq_len?: number
}

export interface TrainingRequest {
  file: File
  chunk_size?: number
  seq_len?: number
  batch_size?: number
  epochs?: number
  embedding_dim?: number
  hidden_dim?: number
  num_layers?: number
  patience?: number
}

// API Error Type
export interface APIError {
  detail: string
  status?: number
}

// Generic API caller with error handling
async function apiCall<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  
  const defaultOptions: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  }

  try {
    const response = await fetch(url, defaultOptions)
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error(`API call failed for ${endpoint}:`, error)
    throw error
  }
}

// Text Generation API
export async function generateText(params: GenerateRequest): Promise<GenerateResponse> {
  const requestBody = {
    text: params.text,
    length: params.length || 100,
    seq_len: params.seq_len || 50,
  }

  return apiCall<GenerateResponse>('/generate/', {
    method: 'POST',
    body: JSON.stringify(requestBody),
  })
}

// Model Training API
export async function trainModel(params: TrainingRequest): Promise<TrainingResponse> {
  const formData = new FormData()
  
  // Append the file
  formData.append('file', params.file)
  
  // Append other parameters
  formData.append('chunk_size', String(params.chunk_size || 120))
  formData.append('seq_len', String(params.seq_len || 50))
  formData.append('batch_size', String(params.batch_size || 32))
  formData.append('epochs', String(params.epochs || 30))
  formData.append('embedding_dim', String(params.embedding_dim || 128))
  formData.append('hidden_dim', String(params.hidden_dim || 256))
  formData.append('num_layers', String(params.num_layers || 2))
  formData.append('patience', String(params.patience || 3))

  const url = `${API_BASE_URL}/train/`
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header for FormData, let the browser set it
    })
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error(`API call failed for /train/:`, error)
    throw error
  }
}

// Health Check API
export async function healthCheck(): Promise<{ status: string }> {
  return apiCall<{ status: string }>('/', {
    method: 'GET',
  })
}

// Get API Info
export async function getApiInfo(): Promise<{
  message: string
  endpoints: {
    translate: string
    docs: string
    health: string
    train: string
  }
}> {
  return apiCall('/', {
    method: 'GET',
  })
}