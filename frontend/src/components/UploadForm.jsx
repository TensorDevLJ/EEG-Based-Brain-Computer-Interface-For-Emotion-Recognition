import React, { useState } from 'react'
import { CloudArrowUpIcon } from '@heroicons/react/24/outline'
import axios from 'axios'

export default function UploadForm({ onResult, endpointPath = '/predict', method = 'POST' }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    if (!file) {
      setError('Please select a CSV file')
      return
    }

    const form = new FormData()
    form.append('file', file)

    try {
      setLoading(true)
      // Use the Vite dev server proxy: POST to /api{endpointPath}
      const url = `/api${endpointPath}`
      const res = await axios.post(url, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      })
      onResult(res.data)
    } catch (err) {
      console.error(err)
      const msg = err?.response?.data?.detail || err.message || 'Upload failed'
      setError(msg)
      onResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Features CSV</label>
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => setFile(e.target.files[0])}
            className="mt-1 block w-full text-sm text-gray-700"
          />
        </div>

        <div>
          <button
            type="submit"
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-60"
            disabled={loading}
          >
            <CloudArrowUpIcon className="w-5 h-5" />
            {loading ? 'Uploading...' : `Upload & ${endpointPath.replace('/', '')}`}
          </button>
        </div>

        {error && <div className="text-sm text-red-600">{error}</div>}
      </form>
    </div>
  )
}
