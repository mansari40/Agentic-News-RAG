import type { Message, QueryCostEntry } from "./types"

const MESSAGES_KEY = "timber_messages"
const COSTS_KEY = "timber_costs"

export function loadMessages(): Message[] {
  if (typeof window === "undefined") return []
  try {
    return JSON.parse(localStorage.getItem(MESSAGES_KEY) || "[]")
  } catch { return [] }
}

export function saveMessages(messages: Message[]) {
  if (typeof window === "undefined") return
  localStorage.setItem(MESSAGES_KEY, JSON.stringify(messages.slice(-100)))
}

export function loadCosts(): QueryCostEntry[] {
  if (typeof window === "undefined") return []
  try {
    return JSON.parse(localStorage.getItem(COSTS_KEY) || "[]")
  } catch { return [] }
}

export function saveCosts(costs: QueryCostEntry[]) {
  if (typeof window === "undefined") return
  localStorage.setItem(COSTS_KEY, JSON.stringify(costs))
}

export function clearAll() {
  if (typeof window === "undefined") return
  localStorage.removeItem(MESSAGES_KEY)
  localStorage.removeItem(COSTS_KEY)
}
