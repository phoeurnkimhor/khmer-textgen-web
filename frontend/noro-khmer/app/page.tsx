"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "@/components/sidebar"
import { TranslatorArea } from "@/components/translator-area"
import { useTextGeneration } from "@/hooks/use-text-generation"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Menu, X } from "lucide-react"

export default function Page() {
  const [inputText, setInputText] = useState("")
  const [outputText, setOutputText] = useState("")
  const [length, setLength] = useState(100)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const { generateKhmerText, isLoading, result, error, resetError } = useTextGeneration()

  const handleTranslate = async () => {
    if (!inputText.trim()) return

    resetError()
    const params = { 
      text: inputText, 
      length: length, 
    }
    await generateKhmerText(params)
  }

  // Update output text when result changes
  useEffect(() => {
    if (result) {
      setOutputText(result.generated_text)
    }
  }, [result])

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
      <div className="flex-1 flex flex-col min-w-0">
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
            <h1 className="text-lg font-semibold">Khmer TST</h1>
            <div className="w-8" /> {/* Spacer for center alignment */}
          </div>
        </div>
        
        {error && (
          <Alert className="mx-4 mt-4 border-red-200 bg-red-50">
            <AlertDescription className="text-red-700">
              {error}
            </AlertDescription>
          </Alert>
        )}
        <TranslatorArea
          inputText={inputText}
          outputText={outputText}
          isTranslating={isLoading}
          onInputChange={setInputText}
          onTranslate={handleTranslate}
          length={length}
          onLengthChange={setLength}
        />
      </div>
    </div>
  )
}
