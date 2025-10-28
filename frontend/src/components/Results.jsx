import React, { useState } from 'react'
import { ChartBarIcon, DocumentTextIcon } from '@heroicons/react/24/outline'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

function ProbabilitiesChart({ probs }) {
  if (!probs) return null
  const labels = Object.keys(probs)
  const data = {
    labels,
    datasets: [
      {
        label: 'Probability',
        data: labels.map((k) => probs[k]),
        backgroundColor: 'rgba(99,102,241,0.8)',
      },
    ],
  }

  return <Bar data={data} />
}

function renderAny(value) {
  // Recursively render arbitrary JSON values (primitives, arrays, objects)
  if (value === null || value === undefined) return <span className="text-gray-500">{String(value)}</span>

  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return <span>{String(value)}</span>
  }

  if (Array.isArray(value)) {
    return (
      <ul className="list-disc list-inside">
        {value.map((v, i) => (
          <li key={i} className="mb-1">
            {renderAny(v)}
          </li>
        ))}
      </ul>
    )
  }

  if (typeof value === 'object') {
    const keys = Object.keys(value)
    // If all values are primitives, render a small key/value list
    const allPrimitives = keys.every(
      (k) => value[k] === null || ['string', 'number', 'boolean', 'undefined'].includes(typeof value[k])
    )
    if (allPrimitives) {
      return (
        <div className="space-y-1">
          {keys.map((k) => (
            <div key={k} className="text-sm">
              <span className="font-medium">{k}:</span> <span className="text-gray-700">{String(value[k])}</span>
            </div>
          ))}
        </div>
      )
    }

    // Mixed or nested object: render recursively with indentation
    return (
      <div className="space-y-2">
        {keys.map((k) => (
          <div key={k}>
            <div className="font-medium text-sm">{k}</div>
            <div className="pl-3">{renderAny(value[k])}</div>
          </div>
        ))}
      </div>
    )
  }

  // Fallback: stringify
  return <span>{String(value)}</span>
}

export default function Results({ data, route }) {
  if (!data) {
    return <div className="text-gray-600">No results yet. Upload a CSV or run a query to see output.</div>
  }

  const { probabilities, predicted_class, depression_index, graphs, top_features, explanation_text, confidence, rawResponse, feature_summary } = data

  // Helper flags to know what to show
  const hasPrediction = predicted_class !== undefined && probabilities !== undefined
  const hasExplanation = explanation_text !== undefined || top_features !== undefined
  const hasFeatureSummary = feature_summary !== undefined

  // Sorting state for the top_features table (kept at component level to satisfy hooks rules)
  const [sortCol, setSortCol] = useState(null)
  const [sortDir, setSortDir] = useState('asc')

  // Precompute table-friendly structures for top_features (stable across renders)
  const objItems = Array.isArray(top_features) ? top_features.filter((it) => it && typeof it === 'object' && !Array.isArray(it)) : []
  const uniform = objItems.length > 0 && Array.isArray(top_features) && objItems.length === top_features.length && objItems.every((it) => {
    const keys = Object.keys(it)
    return keys.length === Object.keys(objItems[0]).length && keys.every((k) => k in objItems[0])
  })

  const cols = uniform ? Object.keys(objItems[0]) : []
  const numericCols = cols.map((col) => objItems.every((row) => typeof row[col] === 'number'))

  const sortedItems = (() => {
    if (!uniform) return objItems
    if (!sortCol) return objItems
    const items = [...objItems]
    items.sort((a, b) => {
      const va = a[sortCol]
      const vb = b[sortCol]
      if (va === undefined || va === null) return 1
      if (vb === undefined || vb === null) return -1
      if (typeof va === 'number' && typeof vb === 'number') {
        return (va - vb) * (sortDir === 'asc' ? 1 : -1)
      }
      return String(va).localeCompare(String(vb)) * (sortDir === 'asc' ? 1 : -1)
    })
    return items
  })()

  const toggleSort = (c) => {
    if (sortCol === c) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortCol(c)
      setSortDir('asc')
    }
  }

  return (
    <div className="space-y-6">
      {/* Prediction section - only for endpoints that return class/probabilities */}
      {hasPrediction && (
        <section>
          <h2 className="text-xl font-semibold flex items-center gap-2"><ChartBarIcon className="w-5 h-5 text-indigo-600" /> Prediction</h2>
          <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-gray-50 rounded">
              <div className="text-sm text-gray-500">Predicted class</div>
              <div className="text-2xl font-bold">{predicted_class}</div>
              <div className="text-sm text-gray-600">Confidence: {typeof confidence === 'number' ? (confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
              <div className="text-sm text-gray-600">Depression index: {depression_index !== undefined ? Number(depression_index).toFixed(3) : 'N/A'}</div>
            </div>
            <div className="p-4 bg-gray-50 rounded">
              <h3 className="font-medium mb-2">Probabilities</h3>
              <ProbabilitiesChart probs={probabilities} />
            </div>
          </div>
        </section>
      )}

      {/* Top features / feature summary */}
      {!hasFeatureSummary && (
        <section>
          <h2 className="text-xl font-semibold">Top features</h2>
          <div className="mt-2">
            {Array.isArray(top_features) && top_features.length > 0 ? (
              uniform ? (
                <div className="overflow-auto">
                  <table className="min-w-full divide-y divide-gray-200 text-sm">
                    <thead className="bg-gray-100">
                      <tr>
                        {cols.map((c, i) => (
                          <th
                            key={c}
                            className={`px-3 py-2 text-left font-medium text-gray-700 ${numericCols[i] ? 'w-40' : ''} cursor-pointer select-none`}
                            onClick={() => toggleSort(c)}
                          >
                            <div className="flex items-center gap-2">
                              <span className="truncate">{c}</span>
                              <span className="text-xs text-gray-500">{sortCol === c ? (sortDir === 'asc' ? '▲' : '▼') : ''}</span>
                            </div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-100">
                      {sortedItems.map((row, ri) => (
                        <tr key={ri}>
                          {cols.map((c, ci) => (
                            <td key={c} className={`px-3 py-2 align-top ${numericCols[ci] ? 'text-right tabular-nums' : 'text-left'}`}>
                              {renderAny(row[c])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="space-y-3">
                  {top_features.map((f, i) => (
                    <div key={i} className="p-2 border rounded bg-gray-50">
                      {renderAny(f)}
                    </div>
                  ))}
                </div>
              )
            ) : top_features && typeof top_features === 'object' && Object.keys(top_features).length > 0 ? (
              <div className="p-2 border rounded bg-gray-50">{renderAny(top_features)}</div>
            ) : (
              <div className="text-sm text-gray-500">No top features returned</div>
            )}

            {/* Raw JSON toggle */}
            <details className="mt-3 text-sm">
              <summary className="cursor-pointer text-blue-600">Show raw top_features JSON</summary>
              <pre className="mt-2 p-2 bg-black text-white overflow-auto text-xs rounded">{JSON.stringify(top_features, null, 2)}</pre>
            </details>
          </div>
        </section>
      )}

      {hasExplanation && (
        <section>
          <h2 className="text-xl font-semibold flex items-center gap-2"><DocumentTextIcon className="w-5 h-5 text-yellow-600" /> Explanation</h2>
          <div className="prose max-w-none">
            <p>{explanation_text}</p>
          </div>
        </section>
      )}

      {rawResponse && (
        <section>
          <h2 className="text-xl font-semibold">Raw response</h2>
          <pre className="mt-2 p-3 bg-black text-white overflow-auto text-xs rounded">{JSON.stringify(rawResponse, null, 2)}</pre>
        </section>
      )}

      <section>
        <h2 className="text-xl font-semibold">Graphs</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
          {graphs && typeof graphs === 'object' ? (
            // Render only non-empty graph values
            Object.entries(graphs).filter(([k, v]) => v).map(([k, v]) => (
              <div key={k} className="p-2 bg-white rounded shadow">
                <div className="text-sm text-gray-600 mb-2">{k}</div>
                {/* graphs are expected as base64 image strings */}
                <img src={`data:image/png;base64,${v}`} alt={k} className="w-full h-auto" />
              </div>
            ))
          ) : (
            <div className="text-sm text-gray-500">No graphs returned</div>
          )}
        </div>
      </section>
    </div>
  )
}
