import { API_BASE_URL } from './api'

// Utility to check if the API is reachable
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    return response.ok
  } catch {
    return false
  }
}

// Utility to format API errors for display
export function formatApiError(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  if (typeof error === 'string') {
    return error
  }
  return 'An unexpected error occurred'
}

// Utility to validate text input
export function validateTextInput(text: string): { isValid: boolean; message?: string } {
  if (!text.trim()) {
    return { isValid: false, message: 'Text cannot be empty' }
  }
  
  if (text.length > 1000) {
    return { isValid: false, message: 'Text is too long (max 1000 characters)' }
  }
  
  return { isValid: true }
}

// Utility to validate dataset path
export function validateDatasetPath(path: string): { isValid: boolean; message?: string } {
  if (!path.trim()) {
    return { isValid: false, message: 'Dataset path is required' }
  }
  
  if (!path.toLowerCase().endsWith('.csv')) {
    return { isValid: false, message: 'Dataset must be a CSV file' }
  }
  
  return { isValid: true }
}

// Utility to get default training parameters based on model size
export function getTrainingDefaults(modelSize: 'small' | 'medium' | 'large') {
  const defaults = {
    small: {
      embedding_dim: 64,
      hidden_dim: 128,
      num_layers: 1,
      batch_size: 16,
      epochs: 20,
    },
    medium: {
      embedding_dim: 128,
      hidden_dim: 256,
      num_layers: 2,
      batch_size: 32,
      epochs: 30,
    },
    large: {
      embedding_dim: 256,
      hidden_dim: 512,
      num_layers: 3,
      batch_size: 64,
      epochs: 50,
    }
  }
  
  return defaults[modelSize]
}

// Utility to estimate training time (rough calculation)
export function estimateTrainingTime(epochs: number, datasetSize: number): string {
  // Very rough estimation: 1 epoch ≈ 30 seconds per 1000 samples
  const minutesPerEpoch = Math.max(1, Math.ceil(datasetSize / 1000) * 0.5)
  const totalMinutes = epochs * minutesPerEpoch
  
  if (totalMinutes < 60) {
    return `~${totalMinutes} minutes`
  } else {
    const hours = Math.floor(totalMinutes / 60)
    const minutes = totalMinutes % 60
    return `~${hours}h ${minutes}m`
  }
}