import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    const backendUrl = process.env.BACKEND_API_URL || 'http://backend:8800'
    const fullUrl = `${backendUrl}/train/`
    
    console.log(`[train] Calling backend: ${fullUrl}`)
    
    const response = await fetch(fullUrl, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      console.error(`[train] Backend returned ${response.status}`)
      const errorData = await response.text().catch(() => '{}')
      try {
        const parsed = JSON.parse(errorData)
        return NextResponse.json(
          { detail: parsed.detail || 'Backend error' },
          { status: response.status }
        )
      } catch {
        return NextResponse.json(
          { detail: `Backend error: ${response.statusText}` },
          { status: response.status }
        )
      }
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('[train] API route error:', error)
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    )
  }
}
