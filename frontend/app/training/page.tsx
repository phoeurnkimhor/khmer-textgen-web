"use client"

import { useState } from "react"
import { useModelTraining } from "@/hooks/use-model-training"
import { Sidebar } from "@/components/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Menu, Upload } from "lucide-react"

export default function TrainingPage() {
  const { startTraining, isTraining, result, error, resetError, reset, progress } = useModelTraining()
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  
  const [formData, setFormData] = useState({
    chunk_size: 120,
    seq_len: 50,
    batch_size: 32,
    epochs: 30,
    embedding_dim: 128,
    hidden_dim: 256,
    num_layers: 2,
    patience: 3,
  })

  const handleInputChange = (field: string, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: typeof value === 'string' ? (isNaN(Number(value)) ? value : Number(value)) : value
    }))
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const fileName = file.name.toLowerCase()
      const isCSV = fileName.endsWith('.csv') || file.type === 'text/csv' || file.type === 'application/csv'
      const isExcel = fileName.endsWith('.xlsx') || fileName.endsWith('.xls') || 
                       file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
                       file.type === 'application/vnd.ms-excel'
      
      if (isCSV || isExcel) {
        setSelectedFile(file)
      } else {
        alert('Please select a valid CSV or Excel file (.csv, .xlsx, .xls)')
        e.target.value = ''
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    resetError()
    
    if (!selectedFile) {
      return
    }

    await startTraining({ ...formData, file: selectedFile })
  }

  const handleReset = () => {
    reset()
    setSelectedFile(null)
    setFormData({
      chunk_size: 120,
      seq_len: 50,
      batch_size: 32,
      epochs: 30,
      embedding_dim: 128,
      hidden_dim: 256,
      num_layers: 2,
      patience: 3,
    })
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Mobile sidebar overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed lg:relative z-50 lg:z-auto
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        transition-transform duration-200 ease-in-out
      `}>
        <Sidebar onClose={() => setIsSidebarOpen(false)} />
      </div>
      
      {/* Main content */}
      <div className="flex-1 overflow-y-auto min-w-0">
        {/* Mobile header */}
        <div className="lg:hidden border-b border-border p-4">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsSidebarOpen(true)}
              className="h-8 w-8"
            >
              <Menu className="h-4 w-4" />
            </Button>
            <h1 className="text-lg font-semibold">Model Training</h1>
            <div className="w-8" /> {/* Spacer for center alignment */}
          </div>
        </div>
        <div className="container mx-auto p-4 lg:p-6 space-y-6">
          <div className="text-center space-y-2">
            <h1 className="text-2xl lg:text-3xl font-bold">Model Training</h1>
            <p className="text-muted-foreground text-sm lg:text-base">Train a new Khmer text generation model</p>
          </div>

          <div className="space-y-6 lg:grid lg:grid-cols-2 lg:gap-6 lg:space-y-0">
        {/* Training Form */}
        <Card>
          <CardHeader>
            <CardTitle>Training Configuration</CardTitle>
            <CardDescription>
              Configure the parameters for training your model
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="dataset_file">Dataset File *</Label>
                <div className="relative">
                  <Input
                    id="dataset_file"
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileChange}
                    disabled={isTraining}
                    className="cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                  />
                </div>
                {selectedFile && (
                  <div className="flex items-center gap-2 p-2 bg-muted rounded-md">
                    <Upload className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </span>
                  </div>
                )}
                {!selectedFile && (
                  <p className="text-xs text-muted-foreground">
                    Upload a CSV or Excel file (.csv, .xlsx, .xls) containing your training dataset
                  </p>
                )}
              </div>

              <Separator />

              <div className="grid gap-4 grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="chunk_size">Chunk Size</Label>
                  <Input
                    id="chunk_size"
                    type="number"
                    value={formData.chunk_size}
                    onChange={(e) => handleInputChange("chunk_size", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="seq_len">Sequence Length</Label>
                  <Input
                    id="seq_len"
                    type="number"
                    value={formData.seq_len}
                    onChange={(e) => handleInputChange("seq_len", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="batch_size">Batch Size</Label>
                  <Input
                    id="batch_size"
                    type="number"
                    value={formData.batch_size}
                    onChange={(e) => handleInputChange("batch_size", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="epochs">Epochs</Label>
                  <Input
                    id="epochs"
                    type="number"
                    value={formData.epochs}
                    onChange={(e) => handleInputChange("epochs", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="embedding_dim">Embedding Dim</Label>
                  <Input
                    id="embedding_dim"
                    type="number"
                    value={formData.embedding_dim}
                    onChange={(e) => handleInputChange("embedding_dim", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="hidden_dim">Hidden Dim</Label>
                  <Input
                    id="hidden_dim"
                    type="number"
                    value={formData.hidden_dim}
                    onChange={(e) => handleInputChange("hidden_dim", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="num_layers">Number of Layers</Label>
                  <Input
                    id="num_layers"
                    type="number"
                    value={formData.num_layers}
                    onChange={(e) => handleInputChange("num_layers", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="patience">Patience</Label>
                  <Input
                    id="patience"
                    type="number"
                    value={formData.patience}
                    onChange={(e) => handleInputChange("patience", e.target.value)}
                    disabled={isTraining}
                  />
                </div>
              </div>

              <div className="flex gap-2 pt-4">
                <Button type="submit" disabled={isTraining || !selectedFile}>
                  {isTraining ? "Training..." : "Start Training"}
                </Button>
                <Button type="button" variant="outline" onClick={handleReset} disabled={isTraining}>
                  Reset
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>
            <CardDescription>
              Monitor your training progress and results
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {isTraining && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Training in progress...</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} />
              </div>
            )}

            {error && (
              <Alert className="border-red-200 bg-red-50">
                <AlertDescription className="text-red-700">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {result && (
              <div className="space-y-4">
                <Alert className="border-green-200 bg-green-50">
                  <AlertDescription className="text-green-700">
                    {result.message}
                  </AlertDescription>
                </Alert>
                
                <div className="space-y-2">
                  <h4 className="font-semibold">Training Results</h4>
                  <div className="text-sm space-y-1">
                    <div>Test Perplexity: <span className="font-mono">{result.test_perplexity.toFixed(4)}</span></div>
                    <div>Test Accuracy: <span className="font-mono">{(result.test_accuracy * 100).toFixed(2)}%</span></div>
                    <div>Model Path: <span className="font-mono text-xs break-all">{result.model_path}</span></div>
                  </div>
                  
                  {result.public_url && (
                    <div className="pt-2">
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => window.open(result.public_url, '_blank')}
                      >
                        Download Model
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  </div>
</div>
  )
}