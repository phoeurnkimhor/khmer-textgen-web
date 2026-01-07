"use client";

import {
  Sparkles,
  Briefcase,
  Crown,
  MessageCircle,
  GraduationCap,
  Type,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarProps {
  selectedModel: string;
  onModelSelect: (model: string) => void;
}

const models = [
  {
    id: "royal",
    name: "Royal Translator",
    description: "Royalty addressing tone",
    icon: Crown,
  },
  {
    id: "formal",
    name: "Formal Translator",
    description: "Professional business tone",
    icon: Briefcase,
  },
  {
    id: "casual",
    name: "Casual Translator",
    description: "Friendly conversational style",
    icon: MessageCircle,
  },
  {
    id: "simplifier",
    name: "Simplifier",
    description: "Easy-to-understand text",
    icon: Type,
  },
];

export function Sidebar({ selectedModel, onModelSelect }: SidebarProps) {
  return (
    <aside className="w-64 border-r border-border bg-sidebar flex flex-col">
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-2 text-sidebar-foreground">
          <Sparkles className="w-6 h-6 text-blue-500" />
          <span className="text-lg font-semibold">StyleShift AI</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground px-3 py-2">
            SELECT MODEL
          </p>
          {models.map((model) => {
            const Icon = model.icon;
            return (
              <button
                key={model.id}
                onClick={() => onModelSelect(model.id)}
                className={cn(
                  "w-full text-left px-3 py-3 rounded-lg transition-colors",
                  "hover:bg-sidebar-accent",
                  selectedModel === model.id
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-foreground"
                )}
              >
                <div className="flex items-start gap-3">
                  <Icon className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-sm">{model.name}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {model.description}
                    </p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
