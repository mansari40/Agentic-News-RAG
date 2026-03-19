"use client"
import { useState } from "react"
import { ChevronDown, ChevronUp, ExternalLink, Calendar, Tag } from "lucide-react"
import type { Source } from "../lib/types"

export function SourceCard({ source, index }: { source: Source; index: number }) {
  const [open, setOpen] = useState(false)
  const stype = (source.source_type || "?").toUpperCase()
  const title = source.title || "No title"
  const short = title.length > 60 ? title.slice(0, 60) + "…" : title

  const typeColor: Record<string, string> = {
    TAVILY: "badge-blue",
    MEDIASTACK: "badge-amber",
    BASELINE: "badge-green",
  }
  const badgeClass = typeColor[stype] || "badge-gray"

  return (
    <div className="card overflow-hidden animate-fade-in">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3 p-3 text-left hover:bg-surface-2 transition-colors"
      >
        <span className={`badge ${badgeClass} shrink-0`}>{stype}</span>
        <span className="text-sm text-text-2 flex-1 min-w-0 truncate">{short}</span>
        <span className="text-text-3 shrink-0">
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 space-y-2 border-t border-border animate-slide-up">
          {source.relevance_reason && (
            <div className="mt-2 p-2.5 rounded-lg text-xs" style={{ background: "#ecfdf5", borderLeft: "3px solid #34d399", color: "#065f46" }}>
              <span className="font-semibold">Why selected: </span>
              {source.relevance_reason}
            </div>
          )}
          {!source.relevance_reason && (
            <div className="mt-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-text-3">Similarity score</span>
                <span className="text-xs font-mono text-accent">{source.score.toFixed(3)}</span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "#e5e7eb" }}>
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${source.score * 100}%`,
                    background: source.score >= 0.65 ? "#4ade80" : source.score >= 0.35 ? "#f59e0b" : "#f87171"
                  }}
                />
              </div>
            </div>
          )}

          <div className="space-y-1 text-xs text-text-2">
            {source.url && (
              <a href={source.url} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 hover:text-accent transition-colors truncate">
                <ExternalLink size={11} className="shrink-0" />
                <span className="truncate">{source.url}</span>
              </a>
            )}
            {source.published_at && (
              <div className="flex items-center gap-1.5 text-text-3">
                <Calendar size={11} />
                <span>{source.published_at}</span>
              </div>
            )}
            {source.keywords && source.keywords.length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                <Tag size={11} className="text-text-3" />
                {source.keywords.slice(0, 5).map((k, i) => (
                  <span key={i} className="badge badge-gray text-xs">{k}</span>
                ))}
              </div>
            )}
          </div>

          {source.content && (
            <p className="text-xs text-text-3 leading-relaxed border-t border-border pt-2">
              {source.content.slice(0, 400)}{source.content.length > 400 ? "…" : ""}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
