"use client"
import "./globals.css"

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>Timber Market Intelligence</title>
        <meta name="description" content="German Timber Market Intelligence — Baseline and Agentic RAG" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="h-screen overflow-hidden">{children}</body>
    </html>
  )
}
