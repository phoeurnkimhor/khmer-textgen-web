"use client"

import { useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { TranslatorArea } from "@/components/translator-area"

export default function Page() {
  const [selectedModel, setSelectedModel] = useState("formal")
  const [inputText, setInputText] = useState("")
  const [outputText, setOutputText] = useState("")
  const [isTranslating, setIsTranslating] = useState(false)

  const handleTranslate = async () => {
    if (!inputText.trim()) return

    setIsTranslating(true)

    // Simulate API call with delay
    await new Promise((resolve) => setTimeout(resolve, 1500))

    // Mock translation - in production, replace with actual API call
    const mockTranslations: Record<string, string> = {
      formal:
        "This is a formal translation of your input text. The language has been adjusted to maintain a professional tone suitable for business communications and formal settings.",
      casual:
        "Hey! This is a casual version of what you wrote. Super chill and easy to read, perfect for chatting with friends or informal messages.",
      academic:
        "The aforementioned text has been systematically transformed to align with scholarly conventions, demonstrating enhanced lexical sophistication and adherence to academic discourse standards.",
      simplifier:
        "This is a simple version. It uses easy words that anyone can understand. No complex ideas or hard vocabulary here.",
    }

    setOutputText(mockTranslations[selectedModel] || "Translation completed.")
    setIsTranslating(false)
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar selectedModel={selectedModel} onModelSelect={setSelectedModel} />
      <TranslatorArea
        inputText={inputText}
        outputText={outputText}
        isTranslating={isTranslating}
        onInputChange={setInputText}
        onTranslate={handleTranslate}
      />
    </div>
  )
}
