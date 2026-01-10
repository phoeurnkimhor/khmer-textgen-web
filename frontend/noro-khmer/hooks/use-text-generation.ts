import { useState, useCallback } from 'react'
import { generateText, GenerateRequest, GenerateResponse } from '@/lib/api'

interface UseTextGenerationReturn {
  generateKhmerText: (params: GenerateRequest) => Promise<void>
  isLoading: boolean
  result: GenerateResponse | null
  error: string | null
  resetError: () => void
  reset: () => void
}

export function useTextGeneration(): UseTextGenerationReturn {
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const generateKhmerText = useCallback(async (params: GenerateRequest) => {
    try {
      setIsLoading(true)
      setError(null)
      
      const response = await generateText(params)
      setResult(response)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred'
      setError(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const resetError = useCallback(() => {
    setError(null)
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
    setIsLoading(false)
  }, [])

  return {
    generateKhmerText,
    isLoading,
    result,
    error,
    resetError,
    reset,
  }
}