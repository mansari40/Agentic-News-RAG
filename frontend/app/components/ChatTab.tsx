"use client"
import { useState, useRef, useEffect, useCallback } from "react"
import { Send, X, Cpu, Database, RefreshCw, Download } from "lucide-react"
import { streamAgentic, queryBaseline } from "../lib/api"
import { loadMessages, saveMessages, loadCosts, saveCosts } from "../lib/storage"
import { SourceCard } from "./SourceCard"
import { ReasoningTrace } from "./ReasoningTrace"
import { StreamingPanel, resolveStep } from "./StreamingPanel"
import ReactMarkdown from "react-markdown"
import type { Message, QueryResult, SSEEvent, QueryCostEntry } from "../lib/types"
import type { LiveStep, LiveReActEvent } from "./StreamingPanel"

const QUICK_STARTS = [
  ["Timber prices", "What are current German softwood log prices in 2026?"],
  ["Latest news", "What is the latest German timber market news?"],
  ["Bark beetle", "What is the bark beetle impact on German timber supply?"],
  ["Sawmill output", "What is the current sawmill production situation in Germany?"],
  ["EUDR policy", "How will EU EUDR policy affect German timber markets?"],
  ["Price forecast", "What is the German lumber price forecast for the next months?"],
  ["Construction", "How is German housing construction demand affecting timber?"],
  ["Market trends", "What are German timber market trends for 2026?"],
]

const STEP_CONFIDENCE: Record<string, number> = {
  planned: 0.15, researched: 0.40, ranked: 0.45, ranked_passthrough: 0.45,
  verified: 0.70, answer_synthesized: 0.85,
}

export function ChatTab({
  mode,
  initialQuery,
  onQueryConsumed,
  allowedTools,
  dateFrom,
  dateTo,
}: {
  mode: "baseline" | "agentic"
  initialQuery?: string | null
  onQueryConsumed?: () => void
  allowedTools?: string[]
  dateFrom?: string
  dateTo?: string
}) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [running, setRunning] = useState(false)
  const [liveSteps, setLiveSteps] = useState<LiveStep[]>([])
  const [liveReact, setLiveReact] = useState<LiveReActEvent[]>([])
  const [liveConf, setLiveConf] = useState(0)
  const [liveCost, setLiveCost] = useState(0)
  const [liveTokens, setLiveTokens] = useState(0)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    setMessages(loadMessages())
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, liveSteps, liveReact])

  useEffect(() => {
    if (initialQuery) {
      submit(initialQuery)
      onQueryConsumed?.()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const submit = useCallback(async (question: string) => {
    if (!question.trim() || running) return
    const q = question.trim()
    setInput("")
    setRunning(true)
    setLiveSteps([])
    setLiveReact([])
    setLiveConf(0)
    setLiveCost(0)
    setLiveTokens(0)

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q, timestamp: Date.now() }
    setMessages(prev => {
      const next = [...prev, userMsg]
      saveMessages(next)
      return next
    })

    const start = Date.now()

    if (mode === "baseline") {
      try {
        const result = await queryBaseline(q, true, 5, { dateFrom, dateTo })
        result.response_time = (Date.now() - start) / 1000
        const assistantMsg: Message = {
          id: (Date.now() + 1).toString(), role: "assistant",
          content: result.answer, timestamp: Date.now(), result,
        }
        setMessages(prev => {
          const next = [...prev, assistantMsg]
          saveMessages(next)
          return next
        })
        // Save baseline cost entry
        const costEntry: QueryCostEntry = {
          query: q.slice(0, 60),
          cost_usd: result.cost_usd || 0,
          total_tokens: result.total_tokens || 0,
          llm_calls: result.llm_calls || 0,
          confidence: result.confidence || 0,
          query_type: "baseline",
          timestamp: new Date().toLocaleTimeString(),
          mode: "baseline",
          response_time: result.response_time || 0,
        }
        const existingCosts = loadCosts()
        saveCosts([...existingCosts, costEntry])
      } catch (e) {
        const err: Message = { id: (Date.now() + 1).toString(), role: "assistant", content: `Error: ${e}`, timestamp: Date.now() }
        setMessages(prev => { const next = [...prev, err]; saveMessages(next); return next })
      }
      setRunning(false)
      return
    }

    // Agentic streaming
    abortRef.current = new AbortController()
    // eslint-disable-next-line prefer-const
    let finalResult = null as QueryResult | null
    const shownSteps = new Set<string>()

    try {
      await streamAgentic(q, (ev: SSEEvent) => {
        if (ev.type === "step" && ev.step && !shownSteps.has(ev.step)) {
          shownSteps.add(ev.step)
          const { title, desc } = resolveStep(ev.step)
          setLiveSteps(prev => [...prev.map(s => ({ ...s, done: true })), { id: ev.step!, title, desc, done: false }])
          for (const [key, val] of Object.entries(STEP_CONFIDENCE)) {
            if (ev.step!.startsWith(key)) setLiveConf(val)
          }
        }
        if (ev.type === "action") {
          setLiveReact(prev => [...prev, {
            id: `${Date.now()}-${Math.random()}`, type: "action",
            tool: ev.tool, text: (ev.args as Record<string, string>)?.query
          }])
        }
        if (ev.type === "observation") {
          setLiveReact(prev => [...prev, {
            id: `${Date.now()}-${Math.random()}`, type: "observation",
            count: ev.count, summary: ev.summary
          }])
        }
        if (ev.type === "thought" && ev.text) {
          setLiveReact(prev => [...prev, { id: `${Date.now()}-${Math.random()}`, type: "thought", text: ev.text }])
        }
        if (ev.type === "result" && ev.data) {
          finalResult = ev.data
          setLiveConf(finalResult.confidence || 0)
          setLiveCost(finalResult.cost_usd || 0)
          setLiveTokens(finalResult.total_tokens || 0)
        }
        if (ev.type === "complete") {
          setLiveSteps(prev => prev.map(s => ({ ...s, done: true })))
        }
      }, abortRef.current.signal, { allowedTools, dateFrom, dateTo })
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        const err: Message = { id: (Date.now() + 1).toString(), role: "assistant", content: `Error: ${e}`, timestamp: Date.now() }
        setMessages(prev => { const next = [...prev, err]; saveMessages(next); return next })
        setRunning(false)
        setLiveSteps([])
        setLiveReact([])
        return
      }
    }

    if (finalResult) {
      finalResult.response_time = (Date.now() - start) / 1000
      finalResult.mode = "agentic"
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(), role: "assistant",
        content: finalResult.answer, timestamp: Date.now(), result: finalResult,
      }
      setMessages(prev => {
        const next = [...prev, assistantMsg]
        saveMessages(next)
        return next
      })
      // Save cost
      const costEntry: QueryCostEntry = {
        query: q.slice(0, 60),
        cost_usd: finalResult.cost_usd || 0,
        total_tokens: finalResult.total_tokens || 0,
        llm_calls: finalResult.llm_calls || 0,
        confidence: finalResult.confidence || 0,
        query_type: finalResult.query_type || "unknown",
        timestamp: new Date().toLocaleTimeString(),
        mode: "agentic",
        response_time: finalResult.response_time || 0,
        sources_collected: finalResult.researcher_scratchpad
          ?.filter(s => s.type === "observation")
          .reduce((sum, s) => sum + (s.count || 0), 0) || 0,
        sources_verified: finalResult.sources?.length || 0,
        tools_used: finalResult.researcher_scratchpad
          ?.filter(s => s.type === "action" && s.tool)
          .map(s => s.tool!) || [],
      }
      const costs = loadCosts()
      saveCosts([...costs, costEntry])
    }

    setRunning(false)
    setLiveSteps([])
    setLiveReact([])
    setTimeout(() => inputRef.current?.focus(), 100)
  }, [mode, running, allowedTools, dateFrom, dateTo])

  const cancel = () => {
    abortRef.current?.abort()
    setRunning(false)
    setLiveSteps([])
    setLiveReact([])
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      submit(input)
    }
  }

  const exportMarkdown = (msg: Message) => {
    if (!msg.result) return
    const r = msg.result
    const lines = [
      `# German Timber Market Intelligence`,
      `**Query:** ${messages[messages.indexOf(msg) - 1]?.content || ""}`,
      `**Confidence:** ${r.confidence.toFixed(2)} | **Cost:** $${r.cost_usd.toFixed(5)} | **Time:** ${r.response_time.toFixed(2)}s`,
      ``,
      `## Answer`,
      r.answer,
      r.evidence_summary ? `\n## Evidence Summary\n${r.evidence_summary}` : "",
      r.key_facts?.length ? `\n## Key Facts\n${r.key_facts.map(f => `- ${f}`).join("\n")}` : "",
      `\n## Sources (${r.sources.length})`,
      ...r.sources.map((s, i) => `\n### ${i + 1}. ${s.title || "No title"}\n- URL: ${s.url || "—"}\n- Published: ${s.published_at || "—"}${s.relevance_reason ? `\n- Why: ${s.relevance_reason}` : ""}`)
    ].join("\n")
    const blob = new Blob([lines], { type: "text/markdown" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = `timber_report_${Date.now()}.md`
    a.click()
  }

  return (
    <div className="flex flex-col h-full">
      {/* Quick starts */}
      {messages.length === 0 && (
        <div className="px-4 pb-4">
          <p className="text-xs text-text-3 mb-3 uppercase tracking-wider">Quick Start</p>
          <div className="grid grid-cols-4 gap-2">
            {QUICK_STARTS.map(([label, q]) => (
              <button
                key={label}
                onClick={() => submit(q)}
                disabled={running}
                className="btn-ghost text-xs text-left py-2 px-3 border border-border rounded-lg hover:border-border-2 hover:text-text transition-all"
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 space-y-4 pb-2">
        {messages.map((msg, idx) => (
          <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "user" ? (
              <div className="max-w-[75%] px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm"
                style={{ background: "#dcfce7", border: "1px solid #86efac", color: "#065f46" }}>
                {msg.content}
              </div>
            ) : (
              <div className="max-w-[90%] space-y-3 animate-slide-up">
                {/* Mode badge */}
                <div className="flex items-center gap-2">
                  {msg.result?.mode === "agentic"
                    ? <span className="badge badge-green"><Cpu size={10} />Agentic RAG</span>
                    : <span className="badge badge-blue"><Database size={10} />Baseline RAG</span>
                  }
                  {msg.result?.cached && <span className="badge badge-gray">Cached</span>}
                </div>

                {/* Answer */}
                <div className="card p-4 text-sm leading-relaxed text-text"
                  style={{ borderLeft: "3px solid #166534" }}>
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                      strong: ({ children }) => <strong className="font-semibold text-text">{children}</strong>,
                      ol: ({ children }) => <ol className="list-decimal list-outside ml-5 space-y-3 mb-3">{children}</ol>,
                      ul: ({ children }) => <ul className="list-disc list-outside ml-5 space-y-2 mb-3">{children}</ul>,
                      li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                      h1: ({ children }) => <h1 className="text-base font-semibold text-text mb-2">{children}</h1>,
                      h2: ({ children }) => <h2 className="text-sm font-semibold text-text mb-2">{children}</h2>,
                      h3: ({ children }) => <h3 className="text-sm font-medium text-text-2 mb-1">{children}</h3>,
                      code: ({ children }) => <code className="font-mono text-xs px-1.5 py-0.5 rounded" style={{ background: "#f3f4f6", color: "#1f2937" }}>{children}</code>,
                      blockquote: ({ children }) => <blockquote className="border-l-2 border-border pl-3 text-text-2 italic">{children}</blockquote>,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>

                {msg.result && (
                  <>
                    {/* Metrics row */}
                    <div className="flex items-center gap-3 flex-wrap">
                      <span className="text-xs font-mono text-text-3">{msg.result.response_time.toFixed(2)}s</span>
                      <span className="text-xs font-mono text-text-3">{msg.result.sources.length} sources</span>
                      {msg.result.mode === "agentic" && (
                        <>
                          <span className={`text-xs font-mono ${msg.result.confidence >= 0.65 ? "text-accent" : msg.result.confidence >= 0.35 ? "text-amber" : "text-red"}`}>
                            {(msg.result.confidence * 100).toFixed(0)}% conf
                          </span>
                          <span className="text-xs font-mono text-amber">${msg.result.cost_usd.toFixed(5)}</span>
                          <span className="text-xs font-mono text-text-3">{msg.result.llm_calls} calls</span>
                          <span className={`badge text-xs ${msg.result.is_domain_relevant ? "badge-green" : "badge-red"}`}>
                            {msg.result.is_domain_relevant ? "In Scope" : "Out of Scope"}
                          </span>
                        </>
                      )}
                      <button onClick={() => exportMarkdown(msg)} className="btn-ghost text-xs flex items-center gap-1 ml-auto">
                        <Download size={11} /> Export
                      </button>
                    </div>

                    {/* Confidence bar */}
                    {msg.result.mode === "agentic" && (
                      <div className="h-1 rounded-full overflow-hidden bg-surface-3">
                        <div className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: `${msg.result.confidence * 100}%`,
                            background: msg.result.confidence >= 0.65 ? "#4ade80" : msg.result.confidence >= 0.35 ? "#f59e0b" : "#f87171"
                          }} />
                      </div>
                    )}

                    {/* Evidence summary + key facts */}
                    {msg.result.evidence_summary && (
                      <div className="text-xs text-text-2 p-3 rounded-lg leading-relaxed"
                        style={{ background: "#ecfdf5", borderLeft: "3px solid #34d399" }}>
                        <span className="font-semibold text-blue">Evidence: </span>
                        {msg.result.evidence_summary}
                      </div>
                    )}
                    {msg.result.key_facts?.length > 0 && (
                      <div className="p-3 rounded-lg space-y-1"
                        style={{ background: "#d1fae5", borderLeft: "3px solid #34d399" }}>
                        <p className="text-xs font-semibold text-accent mb-1">Key Facts</p>
                        {msg.result.key_facts.map((f, i) => (
                          <p key={i} className="text-xs text-text-2">• {f}</p>
                        ))}
                      </div>
                    )}

                    {/* Reasoning trace */}
                    {msg.result.researcher_scratchpad?.length > 0 && (
                      <ReasoningTrace steps={msg.result.researcher_scratchpad} />
                    )}

                    {/* Sources */}
                    {msg.result.sources.length > 0 && (
                      <div>
                        <p className="text-xs text-text-3 mb-2">
                          Retrieved Sources ({msg.result.sources.length})
                        </p>
                        <div className="space-y-1.5">
                          {msg.result.sources.map((s, i) => (
                            <SourceCard key={i} source={s} index={i} />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Follow-up suggestions */}
                    <div>
                      <p className="text-xs text-text-3 mb-2">Related Questions</p>
                      <div className="flex flex-wrap gap-2">
                        {getFollowUps(messages[idx - 1]?.content || "").map(q => (
                          <button key={q} onClick={() => submit(q)} disabled={running}
                            className="text-xs px-3 py-1.5 rounded-lg border border-border text-text-3 hover:text-text-2 hover:border-border-2 transition-all">
                            {q}
                          </button>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Live streaming display */}
        {running && (
          <div className="max-w-[90%] card p-4 animate-fade-in">
            <StreamingPanel
              steps={liveSteps}
              react={liveReact}
              confidence={liveConf}
              cost={liveCost}
              tokens={liveTokens}
              isRunning={running}
            />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="px-4 pb-4 pt-2">
        <div className="card p-2 flex items-end gap-2"
          style={{ background: "#ffffff", border: "1px solid #d1d5db" }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={`Ask about German timber markets [${mode.toUpperCase()}]`}
            disabled={running}
            rows={1}
            className="flex-1 bg-transparent text-sm text-text outline-none resize-none py-2 px-2 placeholder-text-3 max-h-32"
            style={{ fontFamily: "var(--font-sora)" }}
          />
          {running ? (
            <button onClick={cancel} className="shrink-0 p-2 rounded-lg text-red hover:bg-red-dim transition-colors">
              <X size={16} />
            </button>
          ) : (
            <button
              onClick={() => submit(input)}
              disabled={!input.trim()}
              className="shrink-0 p-2 rounded-lg transition-all duration-200 disabled:opacity-30"
              style={{ background: input.trim() ? "#166534" : undefined, color: input.trim() ? "#4ade80" : "#4a6a4a" }}
            >
              <Send size={16} />
            </button>
          )}
        </div>
        <p className="text-xs text-text-3 mt-1.5 text-center">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  )
}

function getFollowUps(prompt: string): string[] {
  const q = prompt.toLowerCase()
  if (q.includes("price") || q.includes("preis")) return ["What drives German timber price changes?", "Timber price forecast 2026?"]
  if (q.includes("eudr") || q.includes("policy")) return ["EUDR compliance costs for German companies?", "Timeline for EUDR implementation?"]
  if (q.includes("bark") || q.includes("supply")) return ["Import volumes filling supply gap?", "Reforestation progress in Germany?"]
  return ["German timber market outlook 2026?", "Sawmill production capacity 2026?", "EU policy effects on German timber?"]
}
