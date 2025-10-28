import React, { useState } from 'react'

export default function QueryForm({ defaultPath = '/', onResponse }) {
  const [q, setQ] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleGet(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const url = `/api${defaultPath}${q ? `?${q}` : ''}`
      const res = await fetch(url)
      const json = await res.json()
      onResponse(json)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleGet} className="space-y-2">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        className="mt-1 block w-full text-sm text-gray-700 border rounded px-2 py-1"
        placeholder="e.g. key=value&limit=10"
      />
      <div className="flex gap-2">
        <button className="px-3 py-1 bg-green-600 text-white rounded" disabled={loading} type="submit">
          {loading ? 'Loading...' : 'Send GET'}
        </button>
        {error && <div className="text-sm text-red-600">{error}</div>}
      </div>
    </form>
  )
}
