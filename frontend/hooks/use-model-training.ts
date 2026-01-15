import { useState, useCallback } from 'react'
import { trainModel, TrainingRequest, TrainingResponse } from '@/lib/api'

interface UseModelTrainingReturn {
  startTraining: (params: TrainingRequest) => Promise<void>
  isTraining: boolean
  result: TrainingResponse | null
  error: string | null
  resetError: () => void
  reset: () => void
  progress: number
}

export function useModelTraining(): UseModelTrainingReturn {
  const [isTraining, setIsTraining] = useState(false)
  const [result, setResult] = useState<TrainingResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const startTraining = useCallback(async (params: TrainingRequest) => {
    try {
      setIsTraining(true)
      setError(null)
      setProgress(0)
      
      // Simulate progress updates (since the backend doesn't provide real-time progress)
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + Math.random() * 5, 95))
      }, 1000)

      const response = await trainModel(params)
      
      clearInterval(progressInterval)
      setProgress(100)
      setResult(response)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred'
      setError(errorMessage)
      setProgress(0)
    } finally {
      setIsTraining(false)
    }
  }, [])

  const resetError = useCallback(() => {
    setError(null)
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
    setIsTraining(false)
    setProgress(0)
  }, [])

  return {
    startTraining,
    isTraining,
    result,
    error,
    resetError,
    reset,
    progress,
  }
}