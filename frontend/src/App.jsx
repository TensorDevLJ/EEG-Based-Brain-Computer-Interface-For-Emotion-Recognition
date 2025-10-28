import React, { useState, useEffect } from 'react'
import UploadForm from './components/UploadForm'
import Results from './components/Results'
import QueryForm from './components/QueryForm'
import { CloudArrowUpIcon, ArrowPathIcon, ChartBarIcon, DocumentTextIcon } from '@heroicons/react/24/outline'

export default function App() {
  const [result, setResult] = useState(null)
  const [lastRawResponse, setLastRawResponse] = useState(null)

  const endpoints = [
    { method: 'GET', path: '/', label: 'Root' },
    { method: 'POST', path: '/predict', label: 'Predict' },
    { method: 'POST', path: '/analyze', label: 'Analyze' },
    { method: 'GET', path: '/metrics', label: 'Get Metrics' },
  ]
  const [selected, setSelected] = useState(endpoints[0])
  const [previews, setPreviews] = useState({})

  // one-time probe for GET endpoint statuses (removed periodic polling)
  useEffect(() => {
    let mounted = true
    async function refreshAll() {
      const newPreviews = {}
      for (const ep of endpoints) {
        if (ep.method !== 'GET') continue
        const start = performance.now()
        try {
          const res = await fetch(`/api${ep.path}`)
          const time = Math.round(performance.now() - start)
          newPreviews[ep.path] = { status: res.status, time }
        } catch (err) {
          const time = Math.round(performance.now() - start)
          newPreviews[ep.path] = { status: 'ERR', time }
        }
      }
      if (mounted) setPreviews((p) => ({ ...p, ...newPreviews }))
    }

    // run once on mount
    refreshAll()
    return () => {
      mounted = false
    }
  }, [])

  return (
    <div className="min-h-screen flex items-start py-10">
      <div className="container mx-auto px-4">
        <header className="mb-8">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-semibold">EEG Depression Detection — Frontend</h1>
              <p className="text-sm text-gray-600 mt-1">Upload a features CSV and view predictions, graphs and explanations returned by the backend.</p>
            </div>

            <div className="bg-white rounded shadow p-3">
              <div className="flex items-center gap-3">
                {endpoints.map((ep) => {
                  const preview = previews[ep.path]
                  let Icon = ArrowPathIcon
                  if (ep.path === '/predict' || ep.path === '/analyze') Icon = CloudArrowUpIcon
                  if (ep.path === '/metrics') Icon = ChartBarIcon
                  if (ep.path === '/') Icon = DocumentTextIcon
                  return (
                    <div key={ep.path} className="flex flex-col items-start">
                      <button
                        onClick={async () => {
                          setSelected(ep)
                          // perform a quick probe for GET endpoints to show preview (status + timing)
                          if (ep.method === 'GET') {
                            const start = performance.now()
                            try {
                              const res = await fetch(`/api${ep.path}`)
                              const time = Math.round(performance.now() - start)
                              const status = res.status
                              setPreviews((p) => ({ ...p, [ep.path]: { status, time } }))
                              const json = await res.json()
                              setLastRawResponse(json)
                              setResult((r) => ({ ...r, rawResponse: json }))
                            } catch (err) {
                              const time = Math.round(performance.now() - start)
                              setPreviews((p) => ({ ...p, [ep.path]: { status: 'ERR', time } }))
                              setLastRawResponse({ error: String(err) })
                            }
                          }
                        }}
                        className={`flex items-center gap-2 px-3 py-1 rounded text-sm font-medium border ${selected.path === ep.path ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-700'}`}
                        title={`${ep.method} ${ep.path} — ${ep.label}`}
                      >
                        <Icon className={`w-4 h-4 ${selected.path === ep.path ? 'text-white' : ep.method === 'GET' ? 'text-green-600' : 'text-orange-600'}`} />
                        <span className="mr-2 text-xs font-semibold">{ep.method}</span>
                        <span className="truncate">{ep.path}</span>
                      </button>
                      {/* preview badge */}
                          {preview && (
                            <div className="mt-1 text-xs">
                              <span className={`px-1 py-0.5 rounded text-xs border ${preview.status === 'ERR' ? 'bg-red-50 text-red-700 border-red-200' : (String(preview.status).startsWith('2') ? 'bg-green-50 text-green-700 border-green-200' : (String(preview.status).startsWith('3') || String(preview.status).startsWith('4') ? 'bg-yellow-50 text-yellow-800 border-yellow-200' : 'bg-red-50 text-red-700 border-red-200'))}`}>{preview.status}</span>
                              <span className="ml-2 text-gray-500">{preview.time}ms</span>
                            </div>
                          )}
                    </div>
                  )
                })}
              </div>
              <div className="mt-2 text-xs text-gray-600">Selected: <span className="font-medium">{selected.method} {selected.path}</span> — {selected.label}</div>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-1 p-4 bg-white rounded shadow space-y-4">
            {/* GET form for GET endpoints */}
            {selected.method === 'GET' && (
              <div>
                <label className="block text-sm font-medium text-gray-700">Query string (e.g. k=v&k2=v2)</label>
                <QueryForm
                  defaultPath={selected.path}
                  onResponse={(r) => {
                    setLastRawResponse(r)
                    setResult({ ...result, rawResponse: r })
                  }}
                />
              </div>
            )}

            {/* Upload form for POST endpoints (predict / analyze) */}
            {selected.method === 'POST' && (
              <UploadForm
                endpointPath={selected.path}
                onResult={(res) => {
                  setResult(res)
                  setLastRawResponse(res)
                }}
              />
            )}

            <div className="mt-2 text-sm text-gray-500">
              <div>Try another endpoint:</div>
              <div className="flex gap-2 mt-2 flex-wrap">
                {endpoints.map((ep) => (
                  <button
                    key={ep.path}
                    onClick={async () => {
                      setSelected(ep)
                      // auto-fetch GET endpoints
                      if (ep.method === 'GET') {
                        try {
                          const res = await fetch(`/api${ep.path}`)
                          const json = await res.json()
                          setLastRawResponse(json)
                          setResult({ ...result, rawResponse: json })
                        } catch (err) {
                          setLastRawResponse({ error: String(err) })
                        }
                      }
                    }}
                    className={`px-2 py-1 text-xs rounded border ${selected.path === ep.path ? 'bg-indigo-600 text-white' : 'bg-white text-gray-700'}`}
                  >
                    <span className={`mr-1 ${ep.method === 'GET' ? 'text-green-600' : 'text-orange-600'}`}>{ep.method}</span>
                    {ep.path}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="md:col-span-2 p-4 bg-white rounded shadow">
            <Results route={selected.path} data={{ ...result, rawResponse: lastRawResponse }} />
          </div>
        </div>
      </div>
    </div>
  )
}
