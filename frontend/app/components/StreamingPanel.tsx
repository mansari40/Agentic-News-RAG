"use client"
import { CheckCircle, Loader, Zap, Eye, Brain } from "lucide-react"

export interface LiveStep {
  id: string
  title: string
  desc: string
  done: boolean
}

export interface LiveReActEvent {
  id: string
  type: "thought" | "action" | "observation"
  tool?: string
  text?: string
  count?: number
  summary?: string
}

const STEP_LABELS: Record<string, { title: string; desc: string }> = {
  planned:                    { title: "Query Planning",          desc: "Intent classification and retrieval strategy" },
  researched:               { title: "Researcher",   desc: "ReAct agent selecting retrieval tools" },
  ranked:                     { title: "Source Ranking",          desc: "Triaging sources by relevance" },
  ranked_passthrough:         { title: "Source Ranking",          desc: "Within budget — no ranking needed" },
  ranked_no_sources:          { title: "No Sources",              desc: "Nothing retrieved to rank" },
  verified:                   { title: "Evidence Verification",   desc: "LLM selecting relevant sources" },
  verification_no_sources:    { title: "No Sources Found",        desc: "No articles retrieved" },
  out_of_scope:               { title: "Out of Scope",            desc: "Query outside German timber market scope" },
  synthesis_no_sources:       { title: "No Relevant Sources",     desc: "LLM found no relevant evidence" },
  answer_synthesized:         { title: "Answer Synthesized",      desc: "Answer written from verified sources" },
  cache_hit:                  { title: "Cached Response",         desc: "Returning cached answer" },
  conversational:             { title: "Conversational Reply",    desc: "Social message handled directly" },
}

const TOOL_LABELS: Record<string, string> = {
  search_tavily_specialist: "Tavily Specialist",
  search_tavily_web: "Tavily Web",
  search_mediastack: "MediaStack",
  search_baseline: "Baseline DB",
}

export function StreamingPanel({
  steps,
  react,
  confidence,
  cost,
  tokens,
  isRunning,
}: {
  steps: LiveStep[]
  react: LiveReActEvent[]
  confidence: number
  cost: number
  tokens: number
  isRunning: boolean
}) {
  if (steps.length === 0 && react.length === 0 && !isRunning) return null

  return (
    <div className="space-y-3 py-2">
      {/* Pipeline steps */}
      <div className="space-y-1.5">
        {steps.map(step => (
          <div key={step.id} className="flex items-start gap-2.5 animate-slide-up">
            {step.done
              ? <CheckCircle size={14} className="mt-0.5 shrink-0 text-accent" />
              : <Loader size={14} className="mt-0.5 shrink-0 text-text-3 animate-spin" />
            }
            <div>
              <span className={`text-xs font-semibold ${step.done ? "text-accent" : "text-text-2"}`}>
                {step.title}
              </span>
              {step.desc && <span className="text-xs text-text-3"> — {step.desc}</span>}
            </div>
          </div>
        ))}
        {isRunning && steps.length === 0 && (
          <div className="flex items-center gap-2 text-xs text-text-3">
            <Loader size={12} className="animate-spin" />
            <span>Initialising pipeline…</span>
          </div>
        )}
      </div>

      {/* Live ReAct events */}
      {react.length > 0 && (
        <div className="pl-3 border-l-2 border-border space-y-1.5">
          {react.map(ev => (
            <div key={ev.id} className="animate-slide-up">
              {ev.type === "action" && (
                <div className="flex items-center gap-1.5">
                  <Zap size={11} className="text-blue shrink-0" />
                  <span className="text-xs text-blue font-mono">
                    {TOOL_LABELS[ev.tool || ""] || ev.tool}
                  </span>
                  {ev.text && (
                    <span className="text-xs text-text-3 truncate italic"> — {ev.text.slice(0, 60)}</span>
                  )}
                </div>
              )}
              {ev.type === "observation" && (
                <div className="flex items-center gap-1.5">
                  <Eye size={11} className={`shrink-0 ${(ev.count || 0) > 0 ? "text-accent" : "text-amber"}`} />
                  <span className={`text-xs font-mono ${(ev.count || 0) > 0 ? "text-accent" : "text-amber"}`}>
                    {ev.summary?.slice(0, 80)}
                  </span>
                </div>
              )}
              {ev.type === "thought" && (
                <div className="flex items-center gap-1.5">
                  <Brain size={11} className="text-text-3 shrink-0" />
                  <span className="text-xs text-text-3 truncate">{ev.text?.slice(0, 100)}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Live confidence + cost bar */}
      {(confidence > 0 || cost > 0) && (
        <div className="flex items-center gap-4 pt-1">
          {confidence > 0 && (
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-text-3">Live confidence</span>
                <span className="text-xs font-mono text-accent">{(confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1 rounded-full overflow-hidden bg-surface-3">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${confidence * 100}%`,
                    background: confidence >= 0.65 ? "#4ade80" : confidence >= 0.35 ? "#f59e0b" : "#f87171"
                  }}
                />
              </div>
            </div>
          )}
          {cost > 0 && (
            <div className="text-xs font-mono text-amber shrink-0">
              ${cost.toFixed(5)} · {tokens.toLocaleString()} tok
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function resolveStep(step: string): { title: string; desc: string } {
  for (const [key, val] of Object.entries(STEP_LABELS)) {
    if (step.startsWith(key)) return val
  }
  return { title: step, desc: "" }
}
