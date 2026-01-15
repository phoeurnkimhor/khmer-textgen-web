"use client"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { ArrowRight, Copy, X, Loader2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface TranslatorAreaProps {
  inputText: string
  outputText: string
  isTranslating: boolean
  onInputChange: (text: string) => void
  onTranslate: () => void
  length: number
  onLengthChange: (length: number) => void
}

export function TranslatorArea({
  inputText,
  outputText,
  isTranslating,
  onInputChange,
  onTranslate,
  length,
  onLengthChange
}: TranslatorAreaProps) {
  const { toast } = useToast()
  const charCount = inputText.length

  const handleCopy = async () => {
    await navigator.clipboard.writeText(outputText)
    toast({
      title: "Copied to clipboard",
      description: "The translated text has been copied.",
    })
  }

  const handleClear = () => {
    onInputChange("")
  }

  return (
    <main className="flex-1 flex flex-col overflow-hidden">
      <div className="border-b border-border p-6">
        <h1 className="text-3xl font-bold text-balance">Khmer Text Generation</h1>
        <p className="text-muted-foreground mt-2">I can write khmer well btw :) </p>
      </div>

      <div className="flex-1 overflow-hidden p-6">
        <div className="h-full grid md:grid-cols-2 gap-4">
          {/* Input Box */}
          <div className="flex flex-col bg-card border border-border rounded-lg overflow-hidden shadow-sm">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <span className="text-sm font-medium">Input Text</span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">{charCount} characters</span>
                {inputText && (
                  <button
                    onClick={handleClear}
                    className="text-muted-foreground hover:text-foreground transition-colors"
                    aria-label="Clear text"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
            <Textarea
              value={inputText}
              onChange={(e) => onInputChange(e.target.value)}
              placeholder="Enter your text here..."
              className="flex-1 resize-none border-0 focus-visible:ring-0 focus-visible:ring-offset-0 p-4"
            />
          </div>

          {/* Output Box */}
          <div className="flex flex-col bg-card border border-border rounded-lg overflow-hidden shadow-sm">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <span className="text-sm font-medium">Generated Text</span>
              {outputText && (
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
                  aria-label="Copy to clipboard"
                >
                  <Copy className="w-4 h-4" />
                  Copy
                </button>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {isTranslating ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">Translating...</p>
                  </div>
                </div>
              ) : outputText ? (
                <p className="text-foreground leading-relaxed">{outputText}</p>
              ) : (
                <p className="text-muted-foreground italic">Generated text will appear here...</p>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="border-t border-border p-6 space-y-6">
        {/* Parameter Controls */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <Label htmlFor="length-slider" className="text-sm font-medium">
              Length: {length}
            </Label>
            <Slider
              id="length-slider"
              min={10}
              max={500}
              step={10}
              value={[length]}
              onValueChange={(value) => onLengthChange(value[0])}
              className="w-full"
              disabled={isTranslating}
            />
            <p className="text-xs text-muted-foreground">
              Number of characters to generate
            </p>
          </div>
        </div>

        <Button
          onClick={onTranslate}
          disabled={!inputText.trim() || isTranslating}
          size="lg"
          className="w-full md:w-auto md:min-w-[200px]"
        >
          {isTranslating ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Generating
            </>
          ) : (
            <>
              Generate Text
              <ArrowRight className="w-4 h-4 ml-2" />
            </>
          )}
        </Button>
      </div>
    </main>
  )
}
