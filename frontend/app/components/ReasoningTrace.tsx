"use client"
import { Brain, Zap, Eye, AlertCircle } from "lucide-react"
import type { ResearchStep } from "../lib/types"

const TOOL_LABELS: Record<string, string> = {
  search_tavily_specialist: "Tavily Specialist",
  search_tavily_web: "Tavily Web",
  search_mediastack: "MediaStack",
  search_baseline: "Baseline DB",
}

export function ReasoningTrace({ steps }: { steps: ResearchStep[] }) {
  if (!steps || steps.length === 0) return null

  return (
    <details className="group">
      <summary className="cursor-pointer flex items-center gap-2 text-xs text-text-3 hover:text-text-2 transition-colors py-1 select-none">
        <Brain size={12} />
        <span>View reasoning trace ({steps.length} steps)</span>
      </summary>
      <div className="mt-2 space-y-1.5 pl-4 border-l-2 border-border">
        {steps.map((step, i) => (
          <TraceStep key={i} step={step} />
        ))}
      </div>
    </details>
  )
}

function TraceStep({ step }: { step: ResearchStep }) {
  if (step.type === "thought") {
    return (
      <div className="flex gap-2 animate-slide-up">
        <Brain size={12} className="mt-0.5 shrink-0 text-text-3" />
        <div>
          <span className="text-xs font-mono text-text-3">Step {step.step} · Thought</span>
          <p className="text-xs text-text-2 mt-0.5 leading-relaxed">{step.summary}</p>
        </div>
      </div>
    )
  }
  if (step.type === "action") {
    const label = TOOL_LABELS[step.tool || ""] || step.tool || ""
    const query = (step.args as Record<string, string>)?.query || ""
    return (
      <div className="flex gap-2 animate-slide-up">
        <Zap size={12} className="mt-0.5 shrink-0 text-blue" />
        <div>
          <span className="text-xs font-mono text-blue">Step {step.step} · {label}</span>
          {query && <p className="text-xs text-text-3 mt-0.5 italic truncate">{query.slice(0, 80)}</p>}
        </div>
      </div>
    )
  }
  if (step.type === "observation") {
    const hasResults = (step.count || 0) > 0
    return (
      <div className="flex gap-2 animate-slide-up">
        <Eye size={12} className={`mt-0.5 shrink-0 ${hasResults ? "text-accent" : "text-amber"}`} />
        <div>
          <span className={`text-xs font-mono ${hasResults ? "text-accent" : "text-amber"}`}>
            Step {step.step} · {step.count || 0} articles
          </span>
          <p className="text-xs text-text-3 mt-0.5 truncate">{step.summary}</p>
        </div>
      </div>
    )
  }
  if (step.type === "error") {
    return (
      <div className="flex gap-2 animate-slide-up">
        <AlertCircle size={12} className="mt-0.5 shrink-0 text-red" />
        <p className="text-xs text-red">{step.summary}</p>
      </div>
    )
  }
  return null
}
